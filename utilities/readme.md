# 📘 `utilities/README.md`
```markdown
# MCP Discovery Utility

The `mcp_discovery.py` utility dynamically finds and configures all MCP servers
under the `MCP_servers/` folder. It replaces the need to hard-code `server_configs`.

## 🚀 How It Works

- Scans `MCP_servers/` for subpackages.
- Includes any package with a `__main__.py` OR an explicit `ARGS` override in `meta.py`.
- Loads optional metadata (`DISPLAY_NAME`, `DESCRIPTION`) for nicer logs.
- Builds a list of `(display_name, StdioServerParameters)` tuples.

## 🛠 Usage

In your agent (e.g. `agent_with_asyncstack_and_endpoint.py`):

```python
from utilities.mcp_discovery import discover_mcp_servers, describe_server_configs

async def initialize_mcp_environment():
    server_configs = discover_mcp_servers("MCP_servers")
    print("Discovered MCP servers:\n" + describe_server_configs(server_configs))

    for name, params in server_configs:
        # connect sessions, load tools, etc.
        ...
