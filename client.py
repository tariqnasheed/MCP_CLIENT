#!/usr/bin/env python3
"""
ollama-mcp-client: Connects an MCP server to Ollama, enabling tool‑calling.
Uses a synchronous Ollama client in a thread to avoid async conflicts.
"""

import asyncio
import json
from pathlib import Path
import sys
import traceback

# MCP SDK
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Ollama – synchronous client
from ollama import Client as OllamaClient


def load_config() -> dict:
    """Load configuration from config.json."""
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        print(f"Config file not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in config.json: {e}")
            sys.exit(1)


def mcp_tool_to_ollama_tool(mcp_tool) -> dict:
    """Convert an MCP tool definition to the format Ollama expects."""
    parameters = (
        mcp_tool.inputSchema
        if hasattr(mcp_tool, "inputSchema")
        else {"type": "object", "properties": {}}
    )
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": parameters,
        },
    }


# Keywords that indicate the user wants a file‑system operation
FILE_OPERATION_KEYWORDS = [
    "file", "directory", "folder", "read", "write", "list", "create",
    "delete", "move", "edit", "path", "tree", "search", "rename",
    "copy", "content", "contents", "text", "media", "image", "audio"
]


def user_wants_file_operation(user_input: str) -> bool:
    """Return True if the user's input suggests a file‑system operation."""
    lower = user_input.lower()
    return any(word in lower for word in FILE_OPERATION_KEYWORDS)


async def main():
    config = load_config()
    server_command = config["mcp_server_command"]
    model_name = config["ollama_model"]
    ollama_host = config.get("ollama_host", "http://localhost:11434")

    # Synchronous Ollama client – we'll call it via asyncio.to_thread
    ollama_sync = OllamaClient(host=ollama_host)

    server_params = StdioServerParameters(
        command=server_command[0],
        args=server_command[1:],
    )

    print("Starting MCP server...")
    try:
        # Open the MCP stdio connection
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List tools
                tools_result = await session.list_tools()
                mcp_tools = tools_result.tools
                print("\nAvailable tools:")
                for tool in mcp_tools:
                    print(f"  - {tool.name}: {tool.description or '(no description)'}")

                # Convert tools for Ollama
                ollama_tools = [mcp_tool_to_ollama_tool(t) for t in mcp_tools]
                if not ollama_tools:
                    print("No tools available from the MCP server. Exiting.")
                    return

                # Interactive loop
                print("\nEnter your prompt (or 'quit' to exit):")
                messages = []

                # -----------------------------------------------------------------
                # STRICT SYSTEM MESSAGE – prevents unnecessary tool calls
                # -----------------------------------------------------------------
                messages.append({
                    "role": "system",
                    "content": (
                        "You are a strict assistant with access to file‑system tools. "
                        "Rules:\n"
                        "1) ONLY use a tool when the user EXPLICITLY asks to read, write, "
                        "list, create, delete, move, edit, or search for a file or directory.\n"
                        "2) For any other question (trivia, definitions, general knowledge), "
                        "answer directly WITHOUT using any tool.\n"
                        "3) Never invent paths. Never use dummy paths like /dev/null.\n"
                        "If you break these rules, the tool will fail and you will have wasted time."
                    )
                })

                while True:
                    try:
                        user_input = input("> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\nExiting.")
                        break

                    if user_input.lower() == "quit":
                        print("Goodbye!")
                        break
                    if not user_input:
                        continue

                    messages.append({"role": "user", "content": user_input})

                    # -----------------------------------------------------------------
                    # KEYWORD FILTER: only send tools if the user likely wants a file op
                    # -----------------------------------------------------------------
                    if not user_wants_file_operation(user_input):
                        tools_to_send = []   # no tools → model must answer directly
                    else:
                        tools_to_send = ollama_tools

                    final_answer = False
                    while not final_answer:
                        # Call Ollama in a thread to avoid blocking the MCP event loop
                        try:
                            response = await asyncio.to_thread(
                                ollama_sync.chat,
                                model=model_name,
                                messages=messages,
                                tools=tools_to_send,
                            )
                        except Exception as e:
                            print(f"Ollama error: {e}")
                            if messages and messages[-1]["role"] == "user":
                                messages.pop()
                            break  # back to user prompt

                        message = response["message"]

                        # If the model requested tool calls, execute them
                        if message.get("tool_calls"):
                            messages.append(message)

                            for tool_call in message["tool_calls"]:
                                func_name = tool_call["function"]["name"]

                                # -----------------------------------------------------------------
                                # FIXED ARGUMENT PARSING – handles both dict and string arguments
                                # -----------------------------------------------------------------
                                raw_args = tool_call["function"]["arguments"]
                                if isinstance(raw_args, dict):
                                    func_args = raw_args
                                elif isinstance(raw_args, str):
                                    try:
                                        func_args = json.loads(raw_args)
                                    except json.JSONDecodeError:
                                        func_args = {}
                                else:
                                    func_args = {}

                                # Execute the tool via the MCP session
                                try:
                                    result = await session.call_tool(func_name, func_args)
                                    if result.content and hasattr(result.content[0], "text"):
                                        tool_result = result.content[0].text
                                    else:
                                        tool_result = str(result.content)
                                except Exception as e:
                                    tool_result = f"Error executing tool {func_name}: {e}"

                                print(
                                    f"[Tool call: {func_name}({func_args}) -> "
                                    f"{str(tool_result)[:100]}...]"
                                )

                                messages.append({
                                    "role": "tool",
                                    "content": str(tool_result),
                                    "tool_call_id": tool_call.get("id", ""),
                                    "name": func_name,
                                })

                            # Continue the inner loop – model may need more tool calls
                            continue

                        # No tool calls → final answer
                        final_content = message.get("content", "")
                        print(f"\nAssistant: {final_content}\n")
                        messages.append(message)
                        final_answer = True

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())