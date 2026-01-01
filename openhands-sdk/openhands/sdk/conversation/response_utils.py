"""Utility functions for extracting agent responses from conversation events."""

import re
from collections.abc import Sequence
from typing import Any

from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.event.base import Event
from openhands.sdk.llm.message import content_to_str
from openhands.sdk.tool.builtins.finish import FinishAction, FinishTool


def get_agent_final_response(events: Sequence[Event]) -> str:
    """Extract the final response from the agent.

    An agent can end a conversation in two ways:
    1. By calling the finish tool
    2. By returning a text message with no tool calls

    Args:
        events: List of conversation events to search through.

    Returns:
        The final response message from the agent, or empty string if not found.
    """
    # Find the last finish action or message event from the agent
    for event in reversed(events):
        # Case 1: finish tool call
        if (
            isinstance(event, ActionEvent)
            and event.source == "agent"
            and event.tool_name == FinishTool.name
        ):
            # Extract message from finish tool call
            if event.action is not None and isinstance(event.action, FinishAction):
                return event.action.message
            else:
                break
        # Case 2: text message with no tool calls (MessageEvent)
        elif isinstance(event, MessageEvent) and event.source == "agent":
            text_parts = content_to_str(event.llm_message.content)
            return "".join(text_parts)
    return ""

def parse_structured_response(text: str, expected_output: type[Any]) -> Any:
    """Parse a structured response from agent text.

    Args:
        text: The text response from the agent.
        expected_output: The Pydantic model class to validate against.

    Returns:
        An instance of expected_output.

    Raises:
        ValueError: If parsing fails.
    """
    # 1. Try to find markdown block
    json_block_pattern = re.compile(
        r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
    )
    match = json_block_pattern.search(text)
    if match:
        json_str = match.group(1)
    else:
        # 2. Try to find the first `{` and the last `}`
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start : end + 1]
        else:
            json_str = text.strip()

    try:
        return expected_output.model_validate_json(json_str)
    except Exception as e:
        # If the heuristic failed, maybe text.strip was better?
        if json_str != text.strip():
            try:
                return expected_output.model_validate_json(text.strip())
            except Exception:
                pass
        raise ValueError(f"Failed to parse expected output: {e}. Content: {text}")

