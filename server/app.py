"""
server/app.py — OpenEnv required entry point.

The openenv validator requires:
  - A main() function
  - An `if __name__ == '__main__'` guard that calls main()
  - [project.scripts] pointing to server.app:main

Usage:
    python -m server.app
    uvicorn env.server:app --host 0.0.0.0 --port 7860
"""

import uvicorn
from env.server import app  # noqa: F401  (re-export for ASGI runners)


def main() -> None:
    """Launch the SQLDebugEnv FastAPI server."""
    uvicorn.run(
        "env.server:app",
        host="0.0.0.0",
        port=7860,
        workers=1,
    )


if __name__ == "__main__":
    main()

