# Copyright (c) 2025 bakajikara
#
# This file is licensed under the MIT License (MIT).
# See the LICENSE file in the project root for the full license text.

"""
Command-line interface for sfutils.

Provides subcommands:
- compile: build a SoundFont from an expanded folder
- decompile: expand a SoundFont into a folder

This module exposes small entry functions that can be used as console_scripts
entry points (they must be callables taking no arguments).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


from .compiler import SF2Compiler
from .decompiler import SoundFontDecompiler


def _build_root_parser():
    p = argparse.ArgumentParser(prog="sfutils", description="sfutils command-line tool")
    sub = p.add_subparsers(dest="command", required=True)

    c_compile = sub.add_parser("compile", help="Compile a directory into a SoundFont file")
    c_compile.add_argument("input_dir", help="Input directory with info.json, samples/, instruments/, presets/")
    c_compile.add_argument("output_sf", help="Output SoundFont file path")

    c_decompile = sub.add_parser("decompile", help="Decompile a SoundFont file into a directory")
    c_decompile.add_argument("sf_path", help="Input SoundFont file path")
    c_decompile.add_argument("output_dir", help="Output directory to create")

    return p


def main(argv=None):
    """
    Generic entry point for `python -m sfutils` or package-level CLI.

    Returns exit code (0 on success).
    """
    argv = list(argv) if argv is not None else None
    parser = _build_root_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "compile":
            inp = Path(args.input_dir)
            out = args.output_sf
            compiler = SF2Compiler(inp, out)
            compiler.compile()
        elif args.command == "decompile":
            sf = Path(args.sf_path)
            outdir = Path(args.output_dir)
            decompiler = SoundFontDecompiler(sf, outdir)
            decompiler.decompile()
        else:
            parser.print_help()
            return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
