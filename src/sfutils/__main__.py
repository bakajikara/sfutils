# Copyright (c) 2025 bakajikara
#
# This file is licensed under the MIT License (MIT).
# See the LICENSE file in the project root for the full license text.

"""
Make the package runnable with `python -m sfutils`.

It delegates to `sfutils.cli.main` so the same parser/behavior is used.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
