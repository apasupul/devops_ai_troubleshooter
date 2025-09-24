📘 MCP_servers/README.md
# MCP Servers Template

This folder contains all MCP server implementations. Each server lives in its own subpackage
(e.g. `jenkins_mcp/`, `kb_mcp/`, etc.), and can be discovered and launched dynamically.

## 📂 Structure



MCP_servers/
├─ init.py
├─ _template/ # cookie-cutter template for new servers
│ ├─ __init__.py
│ ├─ __main__.py
│ ├─ core.py
│ └─ meta.py
├─ jenkins_mcp/
│ ├─ __init__.py
│ ├─ __main__.py
│ ├─ core.py
│ └─ meta.py
└─ ...


## ✨ Creating a New MCP Server

1. **Copy the template**
   ```bash
   cp -r MCP_servers/_template MCP_servers/my_new_server


Edit meta.py

DISPLAY_NAME = "My New MCP Server"
DESCRIPTION = "Does cool custom things"
# COMMAND = "python"  # optional
# ARGS = ["-m", "MCP_servers.my_new_server"]  # optional override


Edit core.py

Replace the dummy loop with your actual MCP logic.

Expose a run_server() function.

import asyncio

async def serve():
    print("🚀 My New MCP Server started")
    # TODO: Replace with your server logic
    await asyncio.sleep(999999)

def run_server():
    asyncio.run(serve())


Run your server standalone (for testing)

python -m MCP_servers.my_new_server


You should see your log output.

Restart your main agent

The dynamic discovery utility will automatically detect the new server.

No need to manually edit server_configs.

📌 Notes

Every server must have:

__init__.py → makes it a package.

__main__.py → entrypoint, usually calls run_server() from core.py.

meta.py is optional but recommended for friendly names and descriptions.

If a server does not have a __main__.py, you must define ARGS in meta.py.
