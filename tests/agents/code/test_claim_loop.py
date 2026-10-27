import dataclasses
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from auditor.agents.code import check
from auditor.agents.code.claim_loop import LIMIT_MESSAGE, ClaimVerdict
from auditor.infra.context import RunContext
from auditor.infra.llm import CallLimitReachedError, StructuredLLM
from auditor.infra.scope import Budget, Scope, enter_scope
from auditor.infra.usage import UsageLog
from auditor.planner.code_claims import CLAIM_KEY, ClaimPlan, claim_tasks
from auditor.planner.tasks import PlannerOptions, plan_paper
from auditor.schemas import (
    AuditTask,
    DocumentAnchor,
    ErrorClass,
    EvidenceKind,
    ParsedPaper,
    RepositoryRef,
    SentenceAnchor,
    SettingTarget,
    TaskOutcome,
)
from auditor.schemas.examples import example_paper
from auditor.testing import ScriptedChatModel, offline_context, tool_reply

CLAIM = "We keep the official test split held out and select checkpoints on the validation set."
ARCH = "Our encoder uses a residual connection around every attention block of the network."
TRAIN = (
    "from datasets import load_dataset\n"
    "\n"
    "\n"
    "def main():\n"
    '    data = load_dataset("glue", "sst2")\n'
    '    model = train(data["train"])\n'
    '    best = select(model, data["test"])\n'
    '    print(evaluate(best, data["test"]))\n'
)


def claim_paper() -> ParsedPaper:
    data = example_paper().model_dump()
    data["sentences"][4]["text"] = CLAIM
    data["sentences"][5]["text"] = ARCH
    return ParsedPaper.model_validate(data)


def repo(tmp_path: Path) -> RepositoryRef:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.py").write_text(TRAIN, encoding="utf-8")
    (root / "README.md").write_text("# SST-2 demo\n", encoding="utf-8")
    return RepositoryRef(url=root.resolve().as_uri())


EVALUATION = {
    "sentence_id": "s4",
    "quote": "select checkpoints on the validation set",
    "category": "evaluation",
    "claim": "Checkpoints are selected on the validation split, not the test split.",
}


def context(
    tmp_path: Path, loop: list[AIMessage], **settings: Any
) -> tuple[RunContext, ScriptedChatModel]:
    review = ScriptedChatModel(replies=loop)
    ctx = offline_context(
        tmp_path, paper=claim_paper(), llm=StructuredLLM(ScriptedChatModel(), name="test:main")
    )
    ctx = dataclasses.replace(
        ctx,
        review_llm=StructuredLLM(review, name="test:review"),
        settings=ctx.settings.model_copy(update={"claim_loop_enabled": True, **settings}),
    )
    return ctx, review


async def planned(tmp_path: Path, *entries: dict[str, Any]) -> list[AuditTask]:
    main = ScriptedChatModel(replies=[tool_reply(ClaimPlan, {"claims": list(entries)})])
    llm = StructuredLLM(main, name="test:main")
    return await claim_tasks(claim_paper(), repo(tmp_path), llm, limit=4)


def verdict(**fields: Any) -> AIMessage:
    answer = {
        "verdict": "contradicted",
        "code_path": "train.py",
        "code_start_line": 7,
        "code_end_line": 7,
        "code_does": "selects the checkpoint on the test split.",
        "direct": True,
        "confidence": "high",
        **fields,
    }
    return tool_reply(ClaimVerdict, answer)


def read_train() -> AIMessage:
    return tool_reply("read_repository_file", {"path": "train.py", "start_line": 1})


async def test_the_planner_anchors_claims_on_their_quoted_sentence(tmp_path: Path) -> None:
    invented = {**EVALUATION, "quote": "we never touch the test split at all"}
    unknown = {**EVALUATION, "sentence_id": "s99"}
    (task,) = await planned(tmp_path, EVALUATION, invented, unknown)
    assert task.target == SettingTarget(key=CLAIM_KEY)
    assert task.setting is not None
    assert task.setting.raw_name == "evaluation"
    start = CLAIM.index(EVALUATION["quote"])
    assert task.anchor == SentenceAnchor(
        sentence_id="s4", start=start, end=start + len(EVALUATION["quote"])
    )


async def test_the_planner_call_is_cheap_and_capped(tmp_path: Path) -> None:
    main = ScriptedChatModel(replies=[tool_reply(ClaimPlan, {"claims": []})])
    usage = UsageLog()
    with enter_scope(Scope(run_id="r", usage=usage)):
        llm = StructuredLLM(main, name="test:main")
        assert await claim_tasks(claim_paper(), repo(tmp_path), llm, limit=3) == []
    (call,) = usage.llm_calls
    assert (call.purpose, call.effort) == ("claim-plan", "low")
    assert "at most 3 claims" in str(main.received[0][0].content)


async def test_a_claim_on_a_sentence_the_data_check_covers_is_left_out(tmp_path: Path) -> None:
    (task,) = await planned(tmp_path, EVALUATION)
    covered = task.model_copy(update={"target": SettingTarget(key="data_split")})
    split = {**EVALUATION, "category": "dataset_split"}
    main = ScriptedChatModel(replies=[tool_reply(ClaimPlan, {"claims": [split]})])
    llm = StructuredLLM(main, name="test:main")
    assert await claim_tasks(claim_paper(), repo(tmp_path), llm, limit=4, existing=[covered]) == []


async def test_a_claim_the_code_contradicts_is_a_finding_with_code_evidence(
    tmp_path: Path,
) -> None:
    (task,) = await planned(tmp_path, EVALUATION)
    ctx, review = context(
        tmp_path,
        [tool_reply("search_repository", {"pattern": "test"}), read_train(), verdict()],
    )
    usage = UsageLog()
    with enter_scope(Scope(run_id="r", usage=usage)):
        result = await check(task, ctx)
    assert result.outcome is TaskOutcome.CHECKED
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.METRIC_MISMATCH
    assert finding.anchor == task.anchor
    assert finding.target == SettingTarget(key="eval_metric")
    assert not finding.uncertain
    quote, code = finding.evidence
    assert (quote.kind, quote.text) == (EvidenceKind.QUOTE, CLAIM)
    assert code.kind is EvidenceKind.CODE
    assert (code.path, code.line_start, code.line_end) == ("train.py", 7, 7)
    assert 'data["test"]' in code.text
    # The loop ran on the review model, and saw the tools' output.
    assert len(review.received) == 3
    assert "train.py:7:" in str(review.received[1][-1].content)
    assert {r.purpose for r in usage.llm_calls} == {"claim-loop"}
    assert [r.tool for r in usage.tool_calls] == ["search_repository", "read_repository_file"]


async def test_a_method_claim_is_anchored_on_the_document(tmp_path: Path) -> None:
    arch = {
        "sentence_id": "s5",
        "quote": "residual connection around every attention block",
        "category": "architecture",
        "claim": "Every attention block has a residual connection.",
    }
    (task,) = await planned(tmp_path, arch)
    ctx, _ = context(tmp_path, [read_train(), verdict(direct=False)])
    (finding,) = (await check(task, ctx)).findings
    assert finding.error_class is ErrorClass.METHOD_MISMATCH
    assert finding.anchor == DocumentAnchor()
    assert finding.target == SettingTarget(key="method")
    assert finding.uncertain


async def test_a_claim_the_code_supports_gives_nothing(tmp_path: Path) -> None:
    (task,) = await planned(tmp_path, EVALUATION)
    ctx, _ = context(tmp_path, [read_train(), verdict(verdict="supported")])
    result = await check(task, ctx)
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


async def test_a_claim_the_code_cannot_settle_is_not_a_finding(tmp_path: Path) -> None:
    (task,) = await planned(tmp_path, EVALUATION)
    ctx, _ = context(tmp_path, [verdict(verdict="unsettled", code_does="no selection step")])
    result = await check(task, ctx)
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.findings == ()
    assert result.reason is not None
    assert "no selection step" in result.reason


async def test_a_contradiction_without_lines_it_read_is_dropped(tmp_path: Path) -> None:
    (task,) = await planned(tmp_path, EVALUATION)
    for cited in ({}, {"code_path": "missing.py"}, {"code_start_line": 7, "code_end_line": 90}):
        replies = [verdict(**cited)] if not cited else [read_train(), verdict(**cited)]
        ctx, _ = context(tmp_path, replies)
        result = await check(task, ctx)
        assert result.outcome is TaskOutcome.UNVERIFIABLE
        assert result.findings == ()


async def test_tool_calls_past_the_limit_are_not_run(tmp_path: Path) -> None:
    (task,) = await planned(tmp_path, EVALUATION)
    ctx, review = context(
        tmp_path,
        [tool_reply("search_repository", {"pattern": "test"}), read_train(), verdict()],
        claim_loop_max_tool_calls=1,
    )
    usage = UsageLog()
    with enter_scope(Scope(run_id="r", usage=usage)):
        result = await check(task, ctx)
    assert [r.tool for r in usage.tool_calls] == ["search_repository"]
    refused = review.received[2][-1]
    assert isinstance(refused, ToolMessage)
    assert refused.content == LIMIT_MESSAGE
    # Line 7 was shown by the search, so the contradiction still counts.
    assert [f.error_class for f in result.findings] == [ErrorClass.METRIC_MISMATCH]
    assert "at most 1 tool calls" in str(review.received[0][0].content)


async def test_the_loop_stops_at_its_model_call_limit(tmp_path: Path) -> None:
    (task,) = await planned(tmp_path, EVALUATION)
    searches = [tool_reply("search_repository", {"pattern": f"p{i}"}) for i in range(4)]
    ctx, review = context(tmp_path, searches, claim_loop_max_model_calls=2)
    result = await check(task, ctx)
    assert len(review.received) == 2
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.findings == ()


async def test_the_limits_stay_within_the_task_budget(tmp_path: Path) -> None:
    (task,) = await planned(tmp_path, EVALUATION)
    searches = [tool_reply("search_repository", {"pattern": f"p{i}"}) for i in range(4)]
    ctx, review = context(tmp_path, searches)
    budget = Budget(max_tool_calls=30, max_llm_calls=3)
    budget.llm_calls = 1
    with enter_scope(Scope(run_id="r", budget=budget)):
        result = await check(task, ctx)
    assert len(review.received) == 2
    assert budget.llm_calls == 3
    assert result.outcome is TaskOutcome.UNVERIFIABLE


async def test_without_an_llm_a_claim_is_unverifiable(tmp_path: Path) -> None:
    (task,) = await planned(tmp_path, EVALUATION)
    ctx = offline_context(tmp_path, paper=claim_paper())
    result = await check(task, ctx)
    assert result.outcome is TaskOutcome.UNVERIFIABLE


def claim_loop_tasks(tasks: list[AuditTask]) -> list[AuditTask]:
    return [t for t in tasks if t.target == SettingTarget(key=CLAIM_KEY)]


async def test_the_planner_adds_claim_tasks_only_when_enabled(tmp_path: Path) -> None:
    options = PlannerOptions(repository=repo(tmp_path))
    main = ScriptedChatModel(replies=[tool_reply(ClaimPlan, {"claims": [EVALUATION]})])
    ctx = offline_context(tmp_path, paper=claim_paper(), llm=StructuredLLM(main, name="t:m"))
    off = ctx.settings.model_copy(update={"citation_llm_enabled": False})
    ctx = dataclasses.replace(ctx, settings=off)
    assert claim_loop_tasks(await plan_paper(claim_paper(), ctx, options)) == []
    assert not main.received

    on = dataclasses.replace(ctx, settings=off.model_copy(update={"claim_loop_enabled": True}))
    tasks = claim_loop_tasks(await plan_paper(claim_paper(), on, options))
    assert [t.anchor for t in tasks] == [(await planned(tmp_path, EVALUATION))[0].anchor]
    # Without a repository there is nothing to check the claims against.
    no_repository = PlannerOptions(detect_repository=False)
    assert claim_loop_tasks(await plan_paper(claim_paper(), on, no_repository)) == []


async def test_the_loop_counts_against_the_runs_call_limit(tmp_path: Path) -> None:
    (task,) = await planned(tmp_path, EVALUATION)
    ctx, _ = context(tmp_path, [read_train(), verdict()])
    review = ctx.require_review_llm()
    review.max_calls = 1
    with pytest.raises(CallLimitReachedError):
        await check(task, ctx)
    assert review.calls == 1


async def test_graph_recursion_error_is_unverifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langgraph.errors import GraphRecursionError

    from auditor.agents.code import claim_loop

    (task,) = await planned(tmp_path, EVALUATION)
    ctx, _ = context(tmp_path, [read_train(), verdict()])

    def mock_create_agent(*args: Any, **kwargs: Any) -> Any:
        class FakeAgent:
            async def ainvoke(self, *a: Any, **kw: Any) -> Any:
                raise GraphRecursionError("Recursion limit reached")

        return FakeAgent()

    monkeypatch.setattr(claim_loop, "create_agent", mock_create_agent)
    result = await check(task, ctx)
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "no verdict within" in result.reason


def test_prune_messages_keeps_last_two_turns() -> None:
    from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

    from auditor.agents.code.claim_loop import _prune_messages

    messages: list[AnyMessage] = [
        HumanMessage("start"),
        AIMessage("call 1"),
        ToolMessage("output 1", tool_call_id="call_1"),
        AIMessage("call 2"),
        ToolMessage("output 2", tool_call_id="call_2"),
        AIMessage("call 3"),
        ToolMessage("output 3", tool_call_id="call_3"),
    ]
    pruned = _prune_messages(messages)
    assert pruned[2].content == "[tool output pruned]"
    assert isinstance(pruned[2], ToolMessage)
    assert pruned[2].tool_call_id == "call_1"
    assert pruned[4].content == "output 2"
    assert pruned[6].content == "output 3"


async def test_accounted_prunes_older_tool_messages_in_multi_turn_run(tmp_path: Path) -> None:
    (task,) = await planned(tmp_path, EVALUATION)
    # 3 tool calls followed by final verdict
    replies = [
        tool_reply("search_repository", {"pattern": "eval"}),
        tool_reply("read_repository_file", {"path": "train.py", "start_line": 1}),
        tool_reply("search_repository", {"pattern": "batch"}),
        verdict(verdict="supported"),
    ]
    ctx, review = context(tmp_path, replies, claim_loop_max_model_calls=5)
    result = await check(task, ctx)
    assert result.outcome is TaskOutcome.CHECKED
    # By turn 3 (index 3 in received), the first tool message from turn 0 should be pruned
    assert len(review.received) == 4
    turn3_messages = review.received[3]
    tool_msgs = [m for m in turn3_messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 3
    assert tool_msgs[0].content == "[tool output pruned]"
    assert tool_msgs[1].content != "[tool output pruned]"
    assert tool_msgs[2].content != "[tool output pruned]"
