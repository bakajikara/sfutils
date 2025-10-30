"""
Make the package runnable with `python -m sfutils`.

It delegates to `sfutils.cli.main` so the same parser/behavior is used.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
