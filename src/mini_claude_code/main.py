import json
import os
import time
from dotenv import load_dotenv  # type: ignore
from typing import List, Dict, Any, Tuple
from pathlib import Path
from google import genai  # type: ignore
from google.genai import types, errors  # type: ignore

from .tools import TOOL_REGISTRY, get_tool_str_representation
from .utils import (
    HISTORY_FILE,
    LOG_COLOR,
    RESET_COLOR,
    extract_retry_delay,
    load_conversation,
    save_conversation,
)

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are a helpful coding assistant whose goal is to help complete coding tasks. 
You have access to a series of tools you can execute. Here are the tools you have access to:

{tool_list_str}

When you want to use a tool reply with exactly one line in the format: 'tool: TOOL_NAME({{JSON_ARGS}})' and nothing else.
Use compact single-line JSON with double quotes. After receiving a tool_result(...) message, continue the task.
If no tool is needed, respond normally.
"""

TOOL_COLOR = "\u001b[96m"
RESULT_COLOR = "\u001b[92m"
YOU_COLOR = "\u001b[94m"
ASSISTANT_COLOR = "\u001b[93m"


def create_full_system_prompt() -> str:
    """Builds the complete system prompt by appending dynamic tool descriptions."""
    tool_str_repr = ""
    for tool_name in TOOL_REGISTRY:
        tool_str_repr += "TOOL\n===" + get_tool_str_representation(tool_name)
        tool_str_repr += f"\n{'='*15}\n"
    return SYSTEM_PROMPT.format(tool_list_str=tool_str_repr)


def extract_tool_invocations(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Parses tool invocation requests formatted as 'tool: NAME({...})'."""
    invocations: List[Tuple[str, Dict[str, Any]]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("tool:"):
            continue
        try:
            after = line[len("tool:") :].strip()
            name, rest = after.split("(", 1)
            name = name.strip()
            args, _ = json.JSONDecoder().raw_decode(rest)
            invocations.append((name, args))
        except Exception:
            continue
    return invocations


DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
FALLBACK_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest"
]


def execute_llm_call(conversation: List[Dict[str, str]], max_retries: int = 5, initial_delay: float = 2.0) -> str:
    """Executes a request to Gemini API with automatic fallback and retry handling."""
    contents = []
    for msg in conversation:
        role = "user" if msg["role"] in ["user", "developer"] else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

    # Prepare model fallback candidate list
    models_to_try = [DEFAULT_MODEL] + [m for m in FALLBACK_MODELS if m != DEFAULT_MODEL]

    for model_name in models_to_try:
        delay = initial_delay
        print(f"{LOG_COLOR}[LLM] Using model '{model_name}' ({len(conversation)} messages in context)...{RESET_COLOR}")
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                )
                return response.text
            except errors.APIError as e:
                err_msg = str(e)
                is_404 = "404" in err_msg or getattr(e, "code", None) == 404 or "NOT_FOUND" in err_msg
                is_daily_quota = "GenerateRequestsPerDay" in err_msg or "2h" in err_msg or ("9" in err_msg and "RESOURCE_EXHAUSTED" in err_msg)
                is_429 = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or getattr(e, "code", None) == 429
                is_503 = "503" in err_msg or getattr(e, "code", None) == 503

                # If model is not found (404) or daily quota is exhausted, switch to next fallback model immediately
                if (is_daily_quota or is_404) and model_name != models_to_try[-1]:
                    reason = "404 Not Found" if is_404 else "Daily Quota Exhausted"
                    print(f"\n\033[93m[API Alert] Model '{model_name}' unavailable ({reason}). Switching to fallback model...\033[0m\n")
                    break

                if is_429 or is_503:
                    if attempt < max_retries - 1:
                        wait_time = extract_retry_delay(e) if is_429 else delay
                        status_type = "Rate Limit (429)" if is_429 else "Server Busy (503)"
                        print(f"\n\033[93m[API Warning] {status_type} on '{model_name}'.\033[0m")
                        print(f"\033[90m[Error Details] {err_msg}\033[0m")
                        print(f"\033[93mWaiting retry delay of {wait_time:.1f}s... (Attempt {attempt + 1}/{max_retries})\033[0m\n")
                        time.sleep(wait_time)
                        delay *= 2
                        continue
                raise e
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\n\033[93m[Warning] API call failed on '{model_name}': {e}\033[0m")
                    print(f"\033[93mRetrying in {delay}s...\033[0m\n")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise e
    raise RuntimeError("All configured Gemini models failed or exhausted their quota.")


def run_coding_agent_loop():
    """Main interactive terminal loop for the coding agent."""
    conversation = load_conversation(create_full_system_prompt)
    while True:
        try:
            user_msg = input(f"{YOU_COLOR}You{RESET_COLOR}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_msg:
            continue

        if user_msg.lower() in ["/reset", "/clear"]:
            if Path(HISTORY_FILE).exists():
                Path(HISTORY_FILE).unlink()
            conversation = [{
                "role": "developer",
                "content": create_full_system_prompt()
            }]
            print(f"{LOG_COLOR}[Session] Conversation history reset.{RESET_COLOR}")
            continue

        conversation.append({
            "role": "user",
            "content": user_msg
        })
        save_conversation(conversation)

        while True:
            try:
                assistant_resp = execute_llm_call(conversation)
            except Exception as err:
                print(f"\033[91m[Error] Unable to reach API: {err}\033[0m")
                break
            tool_invocations = extract_tool_invocations(assistant_resp)
            if not tool_invocations:
                print(f"{ASSISTANT_COLOR}Assistant{RESET_COLOR}: {assistant_resp}")
                conversation.append({
                    "role": "assistant",
                    "content": assistant_resp
                })
                save_conversation(conversation)
                break
            print(f"{ASSISTANT_COLOR}Assistant{RESET_COLOR}: {assistant_resp}")
            conversation.append({
                "role": "assistant",
                "content": assistant_resp
            })
            save_conversation(conversation)
            for name, args in tool_invocations:
                print(f"{TOOL_COLOR}[Executing Tool]{RESET_COLOR} {name}({args})")
                tool = TOOL_REGISTRY[name]
                resp = []
                if name == "read_file":
                    resp = tool(args.get("filename", "."))
                elif name == "list_files":
                    resp = tool(args.get("path", "."))
                elif name == "edit_file":
                    resp = tool(args.get("path", "."), args.get("old_str", ""), args.get("new_str", ""))
                log_resp = resp
                if name == "read_file" and isinstance(resp, dict) and "content" in resp:
                    log_resp = {
                        "file_path": resp.get("file_path"),
                        "content_length": len(resp["content"]),
                        "lines": len(resp["content"].splitlines())
                    }
                print(f"{RESULT_COLOR}[Tool Result]{RESET_COLOR} {json.dumps(log_resp)}")
                conversation.append({
                    "role": "user",
                    "content": f"tool_result({json.dumps(resp)})"
                })
                save_conversation(conversation)
            time.sleep(1.0)


if __name__ == "__main__":
    run_coding_agent_loop()
