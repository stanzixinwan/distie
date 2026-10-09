from __future__ import annotations

import asyncio
import logging
import sys

from worker.config import load
from worker.server import serve


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    cfg = load()
    try:
        asyncio.run(serve(cfg))
    except KeyboardInterrupt:
        # A second Ctrl-C during the drain; serve() already logged the shutdown.
        logging.getLogger("worker").warning("forced exit before drain finished")
        sys.exit(130)


if __name__ == "__main__":
    main()
