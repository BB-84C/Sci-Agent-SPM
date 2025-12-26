from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from mcp.server.fastmcp import FastMCP

from .tools.click_anchor import handle as tool_click_anchor
from .tools.fail import handle as tool_fail
from .tools.finish import handle as tool_finish
from .tools.launch_calibrator import handle as tool_launch_calibrator
from .tools.observe import handle as tool_observe
from .tools.set_field import handle as tool_set_field
from .tools.wait_until import handle as tool_wait_until


@dataclass(slots=True)
class McpToolContext:
    step_index: int
    say: str
    signature: Optional[str]
    results: list[dict[str, Any]]


def create_mcp_server(*, agent: Any) -> FastMCP:
    """
    Create an in-process MCP server that exposes the agent's tools.

    Tool implementations delegate to the existing `src/tools/*.py` handlers, but are
    invoked via MCP so the agent can dynamically discover tool schemas with
    `mcp_session.list_tools()`.
    """

    mcp = FastMCP("sci-agent-stm")

    def _ctx() -> McpToolContext:
        ctx = getattr(agent, "_mcp_tool_ctx", None)
        if not isinstance(ctx, McpToolContext):
            raise RuntimeError("MCP tool context not set.")
        return ctx

    def mcp_tool(**kwargs: Any):  # user-requested convenience decorator name
        return mcp.tool(**kwargs)

    @mcp_tool(
        name="observe",
        description="Capture and log ROI screenshots. Use to re-check the UI state.",
        meta={"category": "observation"},
    )
    def observe(rois: Optional[list[str]] = None) -> Literal["continue", "break"]:
        ctx = _ctx()
        action_input: dict[str, Any] = {}
        if rois is not None:
            action_input["rois"] = rois
        return tool_observe(
            agent,
            step_index=ctx.step_index,
            action_input=action_input,
            say=ctx.say,
            signature=ctx.signature,
            results=ctx.results,
        )

    @mcp_tool(
        name="wait_until",
        description="Wait for a UI condition by repeatedly checking a ROI screenshot.",
        meta={"category": "wait"},
    )
    def wait_until(
        roi: str,
        seconds: float,
        max_rounds: int = 10,
        max_total_seconds: int = 6 * 60 * 60,
        reason: Optional[str] = None,
    ) -> Literal["continue", "break"]:
        ctx = _ctx()
        action_input: dict[str, Any] = {
            "roi": roi,
            "seconds": seconds,
            "max_rounds": max_rounds,
            "max_total_seconds": max_total_seconds,
        }
        if reason is not None:
            action_input["reason"] = reason
        return tool_wait_until(
            agent,
            step_index=ctx.step_index,
            action_input=action_input,
            say=ctx.say,
            signature=ctx.signature,
            results=ctx.results,
        )

    @mcp_tool(
        name="click_anchor",
        description="Click a fixed anchor point by name. Optionally capture ROIs after clicking.",
        meta={"category": "action", "side_effects": True},
    )
    def click_anchor(anchor: str, rois: Optional[list[str]] = None) -> Literal["continue", "break"]:
        ctx = _ctx()
        action_input: dict[str, Any] = {"anchor": anchor}
        if rois is not None:
            action_input["rois"] = rois
        return tool_click_anchor(
            agent,
            step_index=ctx.step_index,
            action_input=action_input,
            say=ctx.say,
            signature=ctx.signature,
            results=ctx.results,
        )

    @mcp_tool(
        name="set_field",
        description="Set a UI field by focusing an anchor and typing text, optionally submitting.",
        meta={"category": "action", "side_effects": True},
    )
    def set_field(
        anchor: str,
        typed_text: str,
        submit: Optional[Literal["enter", "tab"]] = "enter",
        rois: Optional[list[str]] = None,
    ) -> Literal["continue", "break"]:
        ctx = _ctx()
        action_input: dict[str, Any] = {"anchor": anchor, "typed_text": typed_text, "submit": submit}
        if rois is not None:
            action_input["rois"] = rois
        return tool_set_field(
            agent,
            step_index=ctx.step_index,
            action_input=action_input,
            say=ctx.say,
            signature=ctx.signature,
            results=ctx.results,
        )

    @mcp_tool(
        name="launch_calibrator",
        description="Launch the ROI/anchor calibrator GUI for the current workspace and stop.",
        meta={"category": "calibration"},
    )
    def launch_calibrator() -> Literal["continue", "break"]:
        ctx = _ctx()
        return tool_launch_calibrator(
            agent,
            step_index=ctx.step_index,
            action_input={},
            say=ctx.say,
            signature=ctx.signature,
            results=ctx.results,
        )

    @mcp_tool(
        name="finish",
        description="Stop the run successfully.",
        meta={"category": "terminal", "terminal": True},
    )
    def finish() -> Literal["continue", "break"]:
        ctx = _ctx()
        return tool_finish(
            agent,
            step_index=ctx.step_index,
            action_input={},
            say=ctx.say,
            signature=ctx.signature,
            results=ctx.results,
        )

    @mcp_tool(
        name="fail",
        description="Stop the run with an error message (may launch the calibrator if appropriate).",
        meta={"category": "terminal", "terminal": True, "error": True},
    )
    def fail(message: str = "Agent failed.") -> Literal["continue", "break"]:
        ctx = _ctx()
        return tool_fail(
            agent,
            step_index=ctx.step_index,
            action_input={"message": message},
            say=ctx.say,
            signature=ctx.signature,
            results=ctx.results,
        )

    return mcp

