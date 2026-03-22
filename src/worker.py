"""DEPRECATED — use ``python -m src work`` instead.

This file is kept only as a backwards-compatible shim so that existing
Dockerfiles / scripts that reference ``src.worker`` continue to work.
"""

import argparse

from src.__main__ import cmd_work


def main() -> None:
    cmd_work(argparse.Namespace())


if __name__ == "__main__":
    main()
