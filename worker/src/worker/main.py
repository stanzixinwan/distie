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
    asyncio.run(serve(cfg))


if __name__ == "__main__":
    main()
