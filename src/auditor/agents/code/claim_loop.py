"""The code and data claim loop: checks one claim a paper makes about its experiments
against the repository, with tools, in a plan, explore and analyse loop.

The planner (`auditor.planner.code_claims`) lists the claims that rest on the released
code or data. Each one becomes a code task, and here a tool-using agent on the review model
searches and reads the repository snapshot until it can say whether the code does what the
claim says. The settings check compares hyperparameter values and the method check sends
the method text with the most related files to the model in one go; the loop instead
looks for the lines one specific claim depends on.

- **Limits.** At most `AUDITOR_CLAIM_LOOP_MAX_TOOL_CALLS` tool calls and
  `AUDITOR_CLAIM_LOOP_MAX_MODEL_CALLS` model calls per claim, and never more than the
  task's budget has left. A tool call past the limit is not run; the model is told to
  answer. Every model call goes through `StructuredLLM.direct_call`, so it counts against
  the run's call and dollar limits and is recorded in the usage log.
- **Evidence gate.** A contradiction becomes a finding only if it cites lines of a file in
  the snapshot that one of its tool calls showed it. The finding quotes the paper
  sentence the planner anchored the claim on, and gives those lines as CODE evidence.
- **No verdict is no finding.** A claim the code cannot settle, or one still open when
  the limits are reached, is unverifiable with its reason. A finding is uncertain unless
  the model calls the cited lines direct evidence with high confidence.

Nothing in the repository is run (D11): the tools only list, search and read its files.
"""

import dataclasses
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from auditor.agents.code.data_checks import paper_quote_text
from auditor.agents.code.method_check import related_files
from auditor.agents.code.tools import make_tools
from auditor.infra.context import RunContext
from auditor.infra.llm import Effort, StructuredLLM, TokenUsage
from auditor.infra.scope import current_scope
from auditor.infra.snapshots import (
    Snapshot,
    SnapshotUnavailableError,
    SourceError,
    fetch_snapshot,
)
from auditor.schemas import (
    AuditTask,
    CandidateFinding,
    DocumentAnchor,
    ErrorClass,
    Evidence,
    EvidenceKind,
    SentenceAnchor,
    SentenceKind,
    SettingTarget,
    SourceKind,
    SourceRef,
    TaskOutcome,
    TaskResult,
)

# The error class of a contradicted claim, and the target its finding reports, by
# category. Dataset, size and evaluation claims are anchored on their sentence like the
# data and settings checks' findings. The rest are method discrepancies, anchored on the
# document with target `method` like the method check's, so they are matched per paper
# (D4).
CATEGORY_CLASSES: dict[str, tuple[ErrorClass, str]] = {
    "dataset_split": (ErrorClass.DATA_SPLIT_MISMATCH, "data_split"),
    "training_examples": (ErrorClass.DATASET_SIZE_MISMATCH, "train_examples"),
    "evaluation": (ErrorClass.METRIC_MISMATCH, "eval_metric"),
    "preprocessing": (ErrorClass.METHOD_MISMATCH, "method"),
    "architecture": (ErrorClass.METHOD_MISMATCH, "method"),
    "training": (ErrorClass.METHOD_MISMATCH, "method"),
}
# A citation of more lines than this does not point at what differs.
MAX_EVIDENCE_LINES = 40
# Output cap of each turn of the loop, thinking included.
TURN_MAX_OUTPUT = 8192
# File paths listed in the first message, so the model need not spend a tool call on it.
MAX_LISTED_FILES = 150
PURPOSE = "claim-loop"
LIMIT_MESSAGE = (
    "The tool call limit for this claim is used up; the call was not run. Give your verdict "
    "now from what you have read."
)

SYSTEM_PROMPT = """You check one claim a research paper makes about its experiments against
the paper's released code and data. Nothing in the repository can be run; you can list,
search and read its files with the tools. Work in three steps:
1. Plan: decide which files, functions, options or names would show what the code does
   about the claim.
2. Explore: search the repository and read those lines. Follow the code from the entry
   point the experiment uses when it matters.
3. Analyse: compare what the lines do with what the claim says.
You have at most {tool_calls} tool calls; use the fewest that settle the claim.

Then answer:
- verdict "contradicted": lines you have read show the code doing something different from
  the claim. Give code_path, code_start_line and code_end_line of those lines (a file you
  read or searched), and say in code_does what they do.
- verdict "supported": the code does what the claim says.
- verdict "unsettled": the code does not show it either way, for example the step is not
  in the repository or depends on a configuration or command line you cannot see.
Set direct to true only if the cited lines themselves show the difference, with no
assumption about which configuration, command line or data file the experiment used.
Confidence is high only if the cited lines clearly contradict the claim. A different
hyperparameter value alone does not count."""


class ClaimVerdict(BaseModel):
    verdict: Literal["contradicted", "supported", "unsettled"]
    code_path: str = ""
    code_start_line: int = Field(default=0, ge=0)
    code_end_line: int = Field(default=0, ge=0)
    code_does: str = ""
    direct: bool = False
    confidence: Literal["low", "medium", "high"] = "low"


@dataclass
class Exploration:
    """What the loop's tool calls did: how many ran, and which file lines they showed."""

    limit: int
    calls: int = 0
    refused: int = 0
    shown: dict[str, set[int]] = field(default_factory=dict)

    def saw(self, path: str, start: int, end: int) -> bool:
        lines = self.shown.get(path, set())
        return any(n in lines for n in range(start, end + 1))


_SEARCH_HIT = re.compile(r"^(?P<path>\S[^\n]*?):(?P<line>\d+): ", re.MULTILINE)
_READ_LINE = re.compile(r"^(?P<line>\d+): ", re.MULTILINE)
_SETTING_HIT = re.compile(r"\((?P<path>[^()\s]+):(?P<line>\d+)\)")


def _note_shown(exploration: Exploration, name: str, args: dict[str, Any], output: str) -> None:
    shown = exploration.shown
    if name == "read_repository_file":
        path = str(args.get("path", ""))
        for match in _READ_LINE.finditer(output):
            shown.setdefault(path, set()).add(int(match["line"]))
    elif name == "search_repository":
        for match in _SEARCH_HIT.finditer(output):
            shown.setdefault(match["path"], set()).add(int(match["line"]))
    elif name == "find_repository_setting":
        for match in _SETTING_HIT.finditer(output):
            shown.setdefault(match["path"], set()).add(int(match["line"]))


def limited_tools(tools: Sequence[BaseTool], exploration: Exploration) -> list[BaseTool]:
    """`tools` sharing one call limit, recording the file lines each call shows."""

    def wrap(tool: BaseTool) -> BaseTool:
        async def run(**kwargs: Any) -> str:
            if exploration.calls >= exploration.limit:
                exploration.refused += 1
                return LIMIT_MESSAGE
            exploration.calls += 1
            output = str(await tool.ainvoke(kwargs))
            _note_shown(exploration, tool.name, kwargs, output)
            return output

        assert tool.args_schema is not None
        return StructuredTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            coroutine=run,
        )

    return [wrap(tool) for tool in tools]


def _usage(response: ModelResponse[Any] | AIMessage) -> TokenUsage:
    messages: list[BaseMessage] = (
        [response] if isinstance(response, AIMessage) else list(response.result)
    )
    total = TokenUsage()
    for message in messages:
        if isinstance(message, AIMessage):
            total += TokenUsage.from_message(message)
    return total


def _prune_messages(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """The messages with the output of all but the last two rounds of tool calls replaced
    by a short note, so each model call does not resend every file read so far."""
    tool_turns: list[list[int]] = []
    current_turn: list[int] = []
    for idx, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            current_turn.append(idx)
        elif current_turn:
            tool_turns.append(current_turn)
            current_turn = []
    if current_turn:
        tool_turns.append(current_turn)

    if len(tool_turns) <= 2:
        return list(messages)

    to_prune = {idx for turn in tool_turns[:-2] for idx in turn}
    pruned: list[AnyMessage] = []
    for idx, msg in enumerate(messages):
        if idx in to_prune:
            pruned.append(
                msg.model_copy(
                    update={
                        "content": "[tool output pruned]",
                        "artifact": None,
                    }
                )
            )
        else:
            pruned.append(msg)
    return pruned


class _Accounted(AgentMiddleware[Any, Any, Any]):
    """Sends each of the agent's model calls through the run's LLM accounting."""

    def __init__(self, llm: StructuredLLM, effort: Effort) -> None:
        super().__init__()
        self.llm = llm
        self.effort = effort

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        pruned_messages = _prune_messages(request.messages)
        if pruned_messages != request.messages:
            request = request.override(messages=pruned_messages)
        return await self.llm.direct_call(
            lambda: handler(request), _usage, effort=self.effort, purpose=PURPOSE
        )


def _limits(ctx: RunContext) -> tuple[int, int]:
    """Tool and model calls this claim may use: the configured limits, within what is
    left of the task's budget."""
    tools = ctx.settings.claim_loop_max_tool_calls
    models = ctx.settings.claim_loop_max_model_calls
    active = current_scope()
    if active is not None:
        budget = active.budget
        if budget.max_tool_calls is not None:
            tools = min(tools, budget.max_tool_calls - budget.tool_calls)
        if budget.max_llm_calls is not None:
            models = min(models, budget.max_llm_calls - budget.llm_calls)
    return max(tools, 0), max(models, 0)


RELATED_FILES = 5


def _passage(ctx: RunContext, sentence_id: str) -> str:
    """The claim's sentence with one sentence of context on each side."""
    body = [s for s in ctx.paper.sentences if s.kind in (SentenceKind.ABSTRACT, SentenceKind.BODY)]
    index = next((i for i, s in enumerate(body) if s.id == sentence_id), None)
    if index is None:
        return ctx.paper.sentence(sentence_id).text
    return " ".join(s.text for s in body[max(0, index - 1) : index + 2])


def _first_message(
    task: AuditTask, ctx: RunContext, snapshot: Snapshot, category: str, claim: str
) -> HumanMessage:
    anchor = task.anchor
    assert isinstance(anchor, SentenceAnchor)
    sentence = ctx.paper.sentence(anchor.sentence_id).text
    files = [f.path for f in snapshot.manifest.files]
    listed = "\n".join(f"  {path}" for path in files[:MAX_LISTED_FILES])
    if len(files) > MAX_LISTED_FILES:
        listed += f"\n  ... and {len(files) - MAX_LISTED_FILES} more (list_repository_files)"
    passage = _passage(ctx, anchor.sentence_id)
    # The files that share the most words with the claim, as a place to start.
    related = related_files(snapshot, f"{claim} {passage}")[:RELATED_FILES]
    hint = "\n".join(f"  {path}" for path in related) or "  (none)"
    return HumanMessage(
        f"Claim ({category}): {claim}\n"
        f"Paper sentence: {sentence}\n"
        f"Passage: {passage}\n\n"
        f"Files most related to the claim:\n{hint}\n\n"
        f"Repository {snapshot.source_id}, files:\n{listed}"
    )


async def explore(
    task: AuditTask,
    ctx: RunContext,
    snapshot: Snapshot,
    *,
    category: str,
    claim: str,
    tool_limit: int,
    model_limit: int,
) -> tuple[ClaimVerdict | None, Exploration]:
    """The verdict of the loop on one claim (None if it stopped without one), and what
    its tool calls did."""
    llm = (
        ctx.require_review_llm() if ctx.settings.claim_loop_model == "review" else ctx.require_llm()
    )
    effort: Effort = ctx.settings.claim_loop_effort
    exploration = Exploration(limit=tool_limit)
    scoped = dataclasses.replace(ctx, repository=task.repository)
    limits: list[AgentMiddleware[Any, Any, Any]] = [
        _Accounted(llm, effort),
        ModelCallLimitMiddleware(run_limit=model_limit, exit_behavior="end"),
    ]
    agent = create_agent(
        llm.model_for(effort, max_output_tokens=TURN_MAX_OUTPUT),
        tools=limited_tools(make_tools(scoped), exploration),
        system_prompt=SYSTEM_PROMPT.format(tool_calls=tool_limit),
        response_format=ToolStrategy(ClaimVerdict),
        middleware=limits,
    )
    config = RunnableConfig(recursion_limit=4 * model_limit + 10)
    message = _first_message(task, ctx, snapshot, category, claim)
    try:
        state = await agent.ainvoke({"messages": [message]}, config)
    except GraphRecursionError:
        return None, exploration
    answer = state.get("structured_response")
    return (answer if isinstance(answer, ClaimVerdict) else None), exploration


def _code_evidence(
    snapshot: Snapshot, verdict: ClaimVerdict, exploration: Exploration
) -> Evidence | str:
    """The cited lines as CODE evidence, or why they do not count."""
    path, start, end = verdict.code_path.strip(), verdict.code_start_line, verdict.code_end_line
    if path not in {f.path for f in snapshot.manifest.files}:
        return f"the cited file {path or '(none)'} is not in the repository snapshot"
    lines = snapshot.read_text(path).splitlines()
    if not 1 <= start <= end <= len(lines):
        return f"the cited lines {start}-{end} are not in {path}"
    if end - start + 1 > MAX_EVIDENCE_LINES:
        return f"the cited span {path}:{start}-{end} is too long to show what differs"
    if not exploration.saw(path, start, end):
        return f"the cited lines {path}:{start}-{end} were never shown to the model"
    return Evidence(
        kind=EvidenceKind.CODE,
        source=SourceRef(
            kind=SourceKind.REPOSITORY, id=snapshot.source_id, url=snapshot.manifest.url
        ),
        text="\n".join(lines[start - 1 : end]),
        path=path,
        line_start=start,
        line_end=end,
    )


async def check_claim(task: AuditTask, ctx: RunContext) -> TaskResult:
    setting, repository = task.setting, task.repository
    if setting is None or repository is None or not isinstance(task.anchor, SentenceAnchor):
        raise ValueError(f"{task.task_id} is not a code claim task")
    if ctx.llm is None:
        return TaskResult.for_task(
            task,
            TaskOutcome.UNVERIFIABLE,
            reason="checking a claim against the code needs an LLM, and none is configured",
        )
    category = setting.raw_name
    claim = setting.raw_value
    tool_limit, model_limit = _limits(ctx)
    if model_limit < 1:
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason="the task has no model calls left"
        )
    try:
        snapshot = await fetch_snapshot(repository, ctx.http, ctx.settings)
    except (SnapshotUnavailableError, SourceError) as exc:
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason=f"the repository could not be fetched: {exc}"
        )
    verdict, exploration = await explore(
        task,
        ctx,
        snapshot,
        category=category,
        claim=claim,
        tool_limit=tool_limit,
        model_limit=model_limit,
    )
    if verdict is None:
        return TaskResult.for_task(
            task,
            TaskOutcome.UNVERIFIABLE,
            reason=(
                f"no verdict within {model_limit} model calls and {tool_limit} tool calls "
                f"({exploration.calls} tool calls made)"
            ),
        )
    if verdict.verdict == "supported":
        return TaskResult.for_task(task, TaskOutcome.CHECKED)
    if verdict.verdict == "unsettled":
        reason = "the code does not settle the claim"
        if verdict.code_does.strip():
            reason += f": {verdict.code_does.strip()[:300]}"
        return TaskResult.for_task(task, TaskOutcome.UNVERIFIABLE, reason=reason)

    evidence = _code_evidence(snapshot, verdict, exploration)
    if isinstance(evidence, str):
        ctx.logger.info("dropping a claim contradiction: %s", evidence)
        return TaskResult.for_task(
            task,
            TaskOutcome.UNVERIFIABLE,
            reason=f"the model reported a contradiction, but {evidence}",
        )
    error_class, target_key = CATEGORY_CLASSES.get(category, (ErrorClass.METHOD_MISMATCH, "method"))
    sentence = ctx.paper.sentence(task.anchor.sentence_id).text
    does = verdict.code_does.strip() or "The cited lines do something else."
    finding = CandidateFinding.for_task(
        task,
        0,
        error_class=error_class,
        summary=f"Paper: {claim} Code: {does}"[:1000],
        anchor=DocumentAnchor() if target_key == "method" else None,
        explanation=(
            f"{category} claim checked against {snapshot.source_id} with "
            f"{exploration.calls} tool calls; {verdict.confidence} confidence"
            + ("" if verdict.direct else "; the cited lines are indirect evidence")
            + "."
        ),
        evidence=[paper_quote_text(task.paper_id, sentence), evidence],
        uncertain=not (verdict.direct and verdict.confidence == "high"),
    ).model_copy(update={"target": SettingTarget(key=target_key)})
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=[finding])
