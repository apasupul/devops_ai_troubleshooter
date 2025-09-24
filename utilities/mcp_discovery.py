# utilities/mcp_discovery.py
import os
import pkgutil
import importlib
from typing import List, Tuple, Optional, Dict, Iterable

# Import your MCP client params type from wherever it lives in your project:
# from mcp.client.stdio import StdioServerParameters
try:
    from mcp.client.stdio import StdioServerParameters  # adjust path if needed
except Exception:  # pragma: no cover
    # Minimal placeholder to avoid import errors if type-only
    class StdioServerParameters:  # type: ignore
        def __init__(self, command: str, args: Iterable[str]):
            self.command = command
            self.args = list(args)

def _parse_csv_env(var_name: str) -> set:
    raw = os.getenv(var_name, "")
    return {x.strip() for x in raw.split(",") if x.strip()}

def _get_meta_for_server(module_path: str) -> Dict[str, Optional[str]]:
    """
    Load optional meta from <module_path>.meta:
      DISPLAY_NAME: str
      DESCRIPTION: str
      COMMAND: str (default 'python')
      ARGS: list[str] (override launch args)
    """
    meta = {
        "display_name": None,
        "description": None,
        "command": "python",
        "args": None,  # default to ["-m", module_path] if not provided
    }
    try:
        m = importlib.import_module(f"{module_path}.meta")
        meta["display_name"] = getattr(m, "DISPLAY_NAME", None)
        meta["description"]  = getattr(m, "DESCRIPTION", None)
        meta["command"]      = getattr(m, "COMMAND", "python") or "python"
        args = getattr(m, "ARGS", None)
        if args is not None:
            meta["args"] = list(args)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        print(f"[discover] Warning: failed to import meta for {module_path}: {e}")
    return meta

def discover_mcp_servers(base_pkg: str = "MCP_servers") -> List[Tuple[str, StdioServerParameters]]:
    """
    Discover subpackages under `base_pkg` that can be launched as MCP servers.

    Filters (env vars):
      MCP_INCLUDE=jira,jenkins
      MCP_EXCLUDE=experimental,temp
    """
    include = _parse_csv_env("MCP_INCLUDE")
    exclude = _parse_csv_env("MCP_EXCLUDE")

    try:
        pkg = importlib.import_module(base_pkg)
    except ModuleNotFoundError as e:
        raise RuntimeError(f"Package '{base_pkg}' not found on PYTHONPATH") from e

    discovered: List[Tuple[str, StdioServerParameters]] = []

    for modinfo in pkgutil.iter_modules(pkg.__path__):
        name = modinfo.name
        if name.startswith("_"):
            continue
        if include and name not in include:
            continue
        if name in exclude:
            continue

        module_path = f"{base_pkg}.{name}"
        has_main = importlib.util.find_spec(f"{module_path}.__main__") is not None
        meta = _get_meta_for_server(module_path)
        display = meta["display_name"] or name

        # Decide launch command/args
        if meta["args"] is not None:
            cmd = meta["command"] or "python"
            launch_args = meta["args"]
        else:
            if not has_main:
                # Skip if no __main__.py and no ARGS override
                continue
            cmd = meta["command"] or "python"
            launch_args = ["-m", module_path]

        params = StdioServerParameters(command=cmd, args=launch_args)
        discovered.append((display, params))

        if meta["description"]:
            print(f"[discover] {display}: {meta['description']}")

    discovered.sort(key=lambda x: x[0].lower())
    return discovered

def describe_server_configs(server_configs: List[Tuple[str, StdioServerParameters]]) -> str:
    """
    Pretty summary for logs/diagnostics.
    """
    lines = []
    for display, params in server_configs:
        lines.append(f" - {display}: {params.command} {' '.join(params.args)}")
    return "\n".join(lines)
