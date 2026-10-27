"""The code agent's capabilities as tools for baseline B1 (decision D7).

B1 gets the same repository snapshot and static scan the code agent uses: list the
files, search them, read a file, look up the values found for a setting, and (in data
analysis mode) read the data report. Each shared function is counted as a tool call
(decision D8). Nothing in the repository is run.
"""

import json
import re

from langchain_core.tools import BaseTool, StructuredTool

from auditor.agents.code.data.report import snapshot_data
from auditor.agents.code.data_checks import static_candidates
from auditor.infra.context import RunContext
from auditor.infra.snapshots import (
    Snapshot,
    SnapshotUnavailableError,
    SourceError,
    fetch_snapshot,
)
from auditor.infra.tools import tracked_tool
from auditor.schemas import RepositorySource

MAX_SEARCH_RESULTS = 15
MAX_READ_LINES = 120
_LINE_CHARS = 200

# Searches reuse the split lines of the last few snapshots instead of rereading them.
_LINES_CACHE_MAX = 4
_snapshot_lines_cache: dict[tuple[str, str], list[tuple[str, list[str]]]] = {}


def _snapshot_lines(snapshot: Snapshot) -> list[tuple[str, list[str]]]:
    key = (str(snapshot.root), snapshot.manifest.revision)
    if key not in _snapshot_lines_cache:
        if len(_snapshot_lines_cache) >= _LINES_CACHE_MAX:
            _snapshot_lines_cache.pop(next(iter(_snapshot_lines_cache)))
        loaded = []
        for file in snapshot.manifest.files:
            try:
                loaded.append((file.path, snapshot.read_text(file.path).splitlines()))
            except (KeyError, OSError, UnicodeDecodeError):
                continue
        _snapshot_lines_cache[key] = loaded
    return _snapshot_lines_cache[key]


async def _snapshot(ctx: RunContext) -> Snapshot | str:
    if ctx.repository is None:
        return "This paper has no code repository to inspect."
    try:
        return await fetch_snapshot(ctx.repository, ctx.http, ctx.settings)
    except (SnapshotUnavailableError, SourceError) as exc:
        return f"The repository could not be fetched: {exc}"


@tracked_tool("list_repository_files")
async def list_repository_files(ctx: RunContext) -> str:
    snapshot = await _snapshot(ctx)
    if isinstance(snapshot, str):
        return snapshot
    manifest = snapshot.manifest
    lines = [f"Repository {snapshot.source_id}", "Downloaded files:"]
    lines += [f"  {f.path} ({f.size} bytes)" for f in manifest.files]
    if manifest.data_files:
        lines.append("Data files (not downloaded; see analyse_data):")
        lines += [
            f"  {d.path}" + (f" ({d.size} bytes)" if d.size else "") for d in manifest.data_files
        ]
    if manifest.incomplete:
        lines.append(f"Incomplete: {'; '.join(manifest.incomplete)}")
    return "\n".join(lines)


@tracked_tool("search_repository")
async def search_repository(ctx: RunContext, pattern: str, max_results: int = 15) -> str:
    snapshot = await _snapshot(ctx)
    if isinstance(snapshot, str):
        return snapshot
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
    limit = max(1, min(max_results, MAX_SEARCH_RESULTS))
    hits: list[str] = []
    for file_path, lines in _snapshot_lines(snapshot):
        for number, line in enumerate(lines, start=1):
            if regex.search(line):
                hits.append(f"{file_path}:{number}: {line.strip()[:_LINE_CHARS]}")
                if len(hits) >= limit:
                    return "\n".join([*hits, f"(stopped after {limit} matches)"])
    return "\n".join(hits) if hits else f"No line matches {pattern!r}."


@tracked_tool("read_repository_file")
async def read_repository_file(
    ctx: RunContext, path: str, start_line: int = 1, end_line: int | None = None
) -> str:
    snapshot = await _snapshot(ctx)
    if isinstance(snapshot, str):
        return snapshot
    try:
        lines = snapshot.read_text(path).splitlines()
    except (KeyError, OSError):
        return f"{path} is not in the repository snapshot; use list_repository_files."
    start = max(1, start_line)
    default_end = start + MAX_READ_LINES - 1
    end = min(
        len(lines),
        end_line if end_line is not None else default_end,
        start + MAX_READ_LINES - 1,
    )
    return "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, end + 1))


@tracked_tool("find_repository_setting")
async def find_repository_setting(ctx: RunContext, key: str) -> str:
    snapshot = await _snapshot(ctx)
    if isinstance(snapshot, str):
        return snapshot
    rows = []
    for candidate in await static_candidates(snapshot):
        setting = candidate.setting
        if setting.key != key:
            continue
        source = setting.source
        assert isinstance(source, RepositorySource)
        context = f" [{setting.context}]" if setting.context else ""
        rows.append(
            f"{setting.value}{context} ({source.path}:{source.line}) from {setting.raw_name}"
        )
    return "\n".join(rows) if rows else f"No value for {key} was found by the static scan."


@tracked_tool("analyse_data")
async def analyse_data(ctx: RunContext) -> str:
    if not ctx.settings.data_analysis_enabled:
        return "Data analysis mode is off for this run, so data files are not read."
    snapshot = await _snapshot(ctx)
    if isinstance(snapshot, str):
        return snapshot
    data = await snapshot_data(snapshot, ctx)
    if data.report is None:
        return f"No data report: {data.reason}"
    return json.dumps(data.report.model_dump(mode="json"), indent=1)


def make_tools(ctx: RunContext) -> list[BaseTool]:
    async def list_files() -> str:
        return await list_repository_files(ctx)

    async def search(pattern: str, max_results: int = 15) -> str:
        return await search_repository(ctx, pattern, max_results)

    async def read(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        return await read_repository_file(ctx, path, start_line, end_line)

    async def find(key: str) -> str:
        return await find_repository_setting(ctx, key)

    async def data() -> str:
        return await analyse_data(ctx)

    return [
        StructuredTool.from_function(
            coroutine=list_files,
            name="list_repository_files",
            description="List the files of the paper's code repository snapshot.",
        ),
        StructuredTool.from_function(
            coroutine=search,
            name="search_repository",
            description=(
                "Search the repository's code, configs and docs for a regular expression "
                "(case-insensitive). Returns path:line: text for each match."
            ),
        ),
        StructuredTool.from_function(
            coroutine=read,
            name="read_repository_file",
            description="Read lines of one repository file, numbered.",
        ),
        StructuredTool.from_function(
            coroutine=find,
            name="find_repository_setting",
            description=(
                "Values the static scan found in the repository for a setting key such as "
                "learning_rate, batch_size, epochs, optimizer, seed, eval_metric, data_split "
                "or test_fraction, with file and line."
            ),
        ),
        StructuredTool.from_function(
            coroutine=data,
            name="analyse_data",
            description=(
                "Rows per split, label counts and test rows that also occur in training "
                "data, read from the dataset's files (only in data analysis mode)."
            ),
        ),
    ]
