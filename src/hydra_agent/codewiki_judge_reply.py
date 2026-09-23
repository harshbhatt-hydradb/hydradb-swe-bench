"""Recover a model reply that placed its JSON in the reasoning channel."""

import json


def promote_reasoning_channel(message):
    """Forward a tool call or judgment the model already wrote in reasoning.

    Some GPT-OSS responses finish with empty content and no tool call. This uses
    that JSON object. Prose without a tool call or a complete judgment is left
    unscored.
    """

    if getattr(message, "content", None) or getattr(message, "tool_calls", None):
        return False
    reasoning = getattr(message, "reasoning", None) or ""
    decoder = json.JSONDecoder()
    payload = None
    for index, character in enumerate(reasoning):
        if character != "{":
            continue
        try:
            found, _ = decoder.raw_decode(reasoning[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(found, dict):
            payload = found
    if payload is None:
        return False
    if isinstance(payload.get("paths"), list):
        from openai.types.chat import ChatCompletionMessageFunctionToolCall
        from openai.types.chat.chat_completion_message_function_tool_call import Function

        arguments = {key: payload[key] for key in ("paths", "query", "search") if key in payload}
        message.tool_calls = [
            ChatCompletionMessageFunctionToolCall(
                id="call_reasoning",
                type="function",
                function=Function(name="docs_navigator", arguments=json.dumps(arguments)),
            )
        ]
        return True
    if (
        payload.get("score") in (0, 1)
        and isinstance(payload.get("reasoning"), str)
        and payload["reasoning"].strip()
        and isinstance(payload.get("evidence"), str)
        and payload["evidence"].strip()
    ):
        message.content = json.dumps(
            {
                key: payload[key]
                for key in ("criteria", "score", "reasoning", "evidence")
                if key in payload
            }
        )
        return True
    return False
