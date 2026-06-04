#!/usr/bin/env python3
"""
ollama-mcp-client: Interactive chat client that connects a local Ollama model
to an MCP server, enabling intelligent file‑system tool calling.
"""

# asyncio lets us run asynchronous code and manage the MCP session's event loop.
import asyncio

# json is needed to parse configuration files and tool‑call arguments.
import json

# pathlib provides an easy, cross‑platform way to locate config.json relative to the script.
from pathlib import Path

# sys gives us access to exit the program if something goes wrong during setup.
import sys

# traceback prints full error information when an unexpected exception occurs.
import traceback

# The MCP Python SDK – ClientSession handles the protocol, StdioServerParameters
# describes how to launch the server process, and stdio_client creates the connection.
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Ollama’s synchronous client. We import it as OllamaClient for clarity.
from ollama import Client as OllamaClient


# ----------------------------------------------------------------------
# Configuration loader
# ----------------------------------------------------------------------
def load_config() -> dict:
    """
    Load the JSON configuration file from the same folder as this script.
    Returns a dictionary with the settings.
    """
    # Determine the path to config.json. __file__ is the path of the current script,
    # .parent gives the directory containing it.
    config_path = Path(__file__).parent / "config.json"

    # If the file doesn't exist, print an error message and stop.
    if not config_path.exists():
        print(f"Config file not found at {config_path}")
        sys.exit(1)

    # Open the file with UTF‑8 encoding (standard for JSON) and parse it.
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in config.json: {e}")
            sys.exit(1)


# ----------------------------------------------------------------------
# Tool format conversion
# ----------------------------------------------------------------------
def mcp_tool_to_ollama_tool(mcp_tool) -> dict:
    """
    Convert an MCP tool definition into the format that Ollama’s /api/chat expects.
    MCP tools have a name, description, and inputSchema; Ollama expects a dict with
    'type': 'function' and a 'function' key containing the name, description, and parameters.
    """
    # Extract the JSON Schema that describes the tool’s arguments.
    # If the tool has no inputSchema, default to an empty object schema.
    parameters = (
        mcp_tool.inputSchema
        if hasattr(mcp_tool, "inputSchema")
        else {"type": "object", "properties": {}}
    )

    # Build and return the Ollama‑compatible tool dictionary.
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": parameters,
        },
    }


# ----------------------------------------------------------------------
# Smart intent detection
# ----------------------------------------------------------------------
# A list of lowercase keywords that indicate the user wants a file‑system operation.
# If any of these words appears in the user’s prompt, we will pass tools to the model.
FILE_OPERATION_KEYWORDS = [
    "file", "directory", "folder", "read", "write", "list", "create",
    "delete", "move", "edit", "path", "tree", "search", "rename",
    "copy", "content", "contents", "text", "media", "image", "audio",
]


def user_wants_file_operation(user_input: str) -> bool:
    """
    Check whether the user’s input likely refers to a file‑system operation.
    Returns True if any of the FILE_OPERATION_KEYWORDS appears in the lowercased input.
    """
    lower = user_input.lower()
    return any(word in lower for word in FILE_OPERATION_KEYWORDS)


# ----------------------------------------------------------------------
# Main asynchronous routine
# ----------------------------------------------------------------------
async def main():
    # Load the configuration so we know which MCP server to launch and which model to use.
    config = load_config()
    server_command = config["mcp_server_command"]  # e.g. ["npx", "-y", ...]
    model_name = config["ollama_model"]
    ollama_host = config.get("ollama_host", "http://localhost:11434")

    # Create a synchronous Ollama client. We will call it inside a separate thread
    # using asyncio.to_thread() to avoid blocking the MCP event loop.
    ollama_sync = OllamaClient(host=ollama_host)

    # Build the MCP server parameters that tell the SDK how to start the server process.
    server_params = StdioServerParameters(
        command=server_command[0],      # the executable, e.g. "npx"
        args=server_command[1:],        # the rest of the arguments
    )

    print("Starting MCP server...")
    try:
        # Use the async context manager stdio_client to start the MCP server process
        # and obtain the read and write streams.
        async with stdio_client(server_params) as (read, write):
            # Wrap the streams in a ClientSession, which manages the MCP handshake.
            async with ClientSession(read, write) as session:
                # Initialize the session – sends the required MCP handshake.
                await session.initialize()

                # Ask the server for its list of tools.
                tools_result = await session.list_tools()
                mcp_tools = tools_result.tools

                # Print the available tools for the user’s information.
                print("\nAvailable tools:")
                for tool in mcp_tools:
                    print(f"  - {tool.name}: {tool.description or '(no description)'}")

                # Convert every MCP tool into the format Ollama understands.
                ollama_tools = [mcp_tool_to_ollama_tool(t) for t in mcp_tools]
                if not ollama_tools:
                    print("No tools available from the MCP server. Exiting.")
                    return

                # Start the interactive chat loop.
                print("\nEnter your prompt (or 'quit' to exit):")
                # messages will hold the conversation history that we send to Ollama.
                messages = []

                # A strict system message that tells the model exactly when to use tools.
                # This prevents it from calling tools on general‑knowledge questions.
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
                    # Ask the user for input. Strip whitespace.
                    try:
                        user_input = input("> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        # Ctrl+D or Ctrl+C – exit gracefully.
                        print("\nExiting.")
                        break

                    # If the user typed 'quit', exit the loop.
                    if user_input.lower() == "quit":
                        print("Goodbye!")
                        break
                    # Ignore completely empty input.
                    if not user_input:
                        continue

                    # Add the user’s message to the conversation history.
                    messages.append({"role": "user", "content": user_input})

                    # -----------------------------------------------------------------
                    # KEYWORD FILTER: decide whether to send tools for this prompt.
                    # If the input doesn't mention any file‑related keywords, we pass
                    # an empty tool list, which forces the model to answer without tools.
                    # -----------------------------------------------------------------
                    if not user_wants_file_operation(user_input):
                        tools_to_send = []          # no tools → direct answer only
                    else:
                        tools_to_send = ollama_tools   # full tool list

                    # This flag becomes True when we get a final answer (no tool calls).
                    final_answer = False
                    while not final_answer:
                        # Call Ollama in a separate thread so we don’t block the MCP event loop.
                        try:
                            response = await asyncio.to_thread(
                                ollama_sync.chat,
                                model=model_name,
                                messages=messages,
                                tools=tools_to_send,
                            )
                        except Exception as e:
                            # If the Ollama call fails (e.g., network error), inform the user.
                            print(f"Ollama error: {e}")
                            # Remove the last user message to keep history clean.
                            if messages and messages[-1]["role"] == "user":
                                messages.pop()
                            break  # go back to waiting for user input

                        # The response contains a 'message' key with the model’s reply.
                        message = response["message"]

                        # If the message contains tool_calls, we need to execute those tools.
                        if message.get("tool_calls"):
                            # Add the assistant’s tool‑call request to the history.
                            messages.append(message)

                            # Process each tool call in the response.
                            for tool_call in message["tool_calls"]:
                                func_name = tool_call["function"]["name"]

                                # -----------------------------------------------------------------
                                # ARGUMENT PARSING – Ollama versions differ in how they return
                                # arguments. Some return a JSON string, others return a dict.
                                # We handle both cases here.
                                # -----------------------------------------------------------------
                                raw_args = tool_call["function"]["arguments"]
                                if isinstance(raw_args, dict):
                                    # Already a dictionary (Ollama v0.4+)
                                    func_args = raw_args
                                elif isinstance(raw_args, str):
                                    # A JSON string – parse it.
                                    try:
                                        func_args = json.loads(raw_args)
                                    except json.JSONDecodeError:
                                        func_args = {}
                                else:
                                    # Unknown format – fallback to empty dict.
                                    func_args = {}

                                # Execute the tool through the MCP session.
                                try:
                                    result = await session.call_tool(func_name, func_args)
                                    # Extract the first text content from the result.
                                    if result.content and hasattr(result.content[0], "text"):
                                        tool_result = result.content[0].text
                                    else:
                                        tool_result = str(result.content)
                                except Exception as e:
                                    tool_result = f"Error executing tool {func_name}: {e}"

                                # Print a short summary of the tool call for the user.
                                print(
                                    f"[Tool call: {func_name}({func_args}) -> "
                                    f"{str(tool_result)[:100]}...]"
                                )

                                # Append the tool result to the conversation as a 'tool' role message.
                                messages.append({
                                    "role": "tool",
                                    "content": str(tool_result),
                                    "tool_call_id": tool_call.get("id", ""),
                                    "name": func_name,
                                })

                            # After processing tool calls, continue the inner loop.
                            # The model may need to make additional tool calls or give a final answer.
                            continue

                        # No tool calls – the model produced its final text response.
                        final_content = message.get("content", "")
                        print(f"\nAssistant: {final_content}\n")
                        # Save the final assistant message for conversation context.
                        messages.append(message)
                        final_answer = True

    except Exception as e:
        # Catch any unexpected error (e.g., MCP server crash, protocol error).
        print(f"Error: {e}")
        traceback.print_exc()


# ----------------------------------------------------------------------
# Program entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Run the asynchronous main function using asyncio.run().
    asyncio.run(main())