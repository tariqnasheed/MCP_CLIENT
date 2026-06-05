```markdown
# Ollama-MCP Client

A Python client that connects an MCP (Model Context Protocol) server to a local Ollama instance. It lets you chat with an LLM that can dynamically call file‑system tools when needed, while answering general questions directly.

## What's New 

- **Smart Tool Filtering** – tools are only provided to the model when your prompt actually mentions file operations (e.g., "list files"). General questions bypass the tool list entirely, eliminating unnecessary tool calls.  
- **Robust Argument Handling** – works correctly with both older Ollama clients (string arguments) and v0.4+ (dict arguments).  
- **Strict System Prompt** – instructs the model to avoid inventing dummy paths and to use tools only when explicitly asked.  
- **Thread‑safe Ollama calls** – uses a synchronous Ollama client inside an async thread to prevent conflicts with the MCP server's internal event loop.

## What it does

1. Starts an MCP server (e.g., the official Filesystem server) as a subprocess.  
2. Lists all available tools (read, write, list, etc.).  
3. Runs an interactive chat loop.  
4. For each user prompt:  
   - If the prompt contains file‑related keywords, the tools are sent to Ollama; otherwise they are omitted.  
   - If Ollama responds with a tool call, the client executes it via the MCP server and feeds the result back to the model.  
   - The final natural‑language answer is displayed.
   
   Defination: MCP is the standardized communication rulebook that connects the LLM's decision, through the Python script, to the MCP Server which actually performs the task.

## Prerequisites

- **Python 3.10+**  
- **Ollama** installed and running locally (default `http://localhost:11434`)  
  Pull a model that supports tool calling, e.g.:  
  ```bash
  ollama pull llama3.2
  ```
- **Node.js** (for `npx` to run the MCP server)  
- The MCP server package used in your config (the example uses `@modelcontextprotocol/server-filesystem`).

## Installation

1. Clone or create the project directory.  
2. Create a virtual environment (recommended):  
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate         # Windows
   ```
3. Install Python dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config.json` to match your setup. Default content:

```json
{
  "mcp_server_command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp/mcp-demo"],
  "ollama_model": "llama3.2",
  "ollama_host": "http://localhost:11434"
}
```

- `mcp_server_command`: command and arguments to launch the MCP server. The example uses the Filesystem server restricted to `/tmp/mcp-demo`.  
- `ollama_model`: name of the Ollama model (must be pulled first).  
- `ollama_host`: Ollama's API endpoint.

Create the allowed directory (if not using `/tmp`):
```bash
mkdir -p /path/to/allowed_dir
```

## Usage

Run the client:
```bash
python client.py
```

You'll see the available tools printed, then an interactive prompt:

```
Enter your prompt (or 'quit' to exit):
```

**Example interactions:**

*General question (no tools sent):*
```
> what is the capital of india
Assistant: The capital of India is New Delhi.
```

*File operation (tools sent):*
```
> list all files in the root directory
[Tool call: list_directory('/') -> File list: ['notes.txt', 'todo.md']]

> read the file notes.txt
[Tool call: read_file('/notes.txt') -> File contents: "Buy milk, eggs, bread"]
```

Type `quit` to exit.

## How the smart filtering works

The client checks the user's input against a list of file‑operation keywords (`file`, `directory`, `read`, `write`, `list`, `create`, etc.).  
- If **no keyword** is detected, `tools=[]` is passed to Ollama, forcing a direct answer.  
- If a **keyword is found**, the full tool list is sent, allowing the model to perform file operations.  

This prevents the model from inventing dummy tool calls for non‑file‑related questions.

## Troubleshooting

- **Model not found** – run `ollama pull <model-name>` first.  
- **Ollama connection refused** – ensure `ollama serve` is running.  
- **MCP server crashes** – check that Node.js is installed and the server command is correct.  
- **Tool call errors** – the server only accesses directories you allow in `config.json`; make sure those directories exist.  

## Project files

```
ollama-mcp-client/
├── .gitignore
├── README.md
├── requirements.txt
├── config.json
└── client.py
```
```