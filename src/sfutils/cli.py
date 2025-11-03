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
import json
import sys
from pathlib import Path


from .compiler import SoundFontCompiler
from .decompiler import SoundFontDecompiler


def _build_root_parser():
    p = argparse.ArgumentParser(prog="sfutils", description="sfutils command-line tool")
    sub = p.add_subparsers(dest="command", required=True)

    c_compile = sub.add_parser("compile", help="Compile a directory into a SoundFont file")
    c_compile.add_argument("input_directory", help="Input directory with info.json, samples/, instruments/, presets/")
    c_compile.add_argument("output_file", nargs="?", help="Output SoundFont file path (default: <input_dir_name>.sf2 or .sf3 based on info.json)")

    c_decompile = sub.add_parser("decompile", help="Decompile a SoundFont file into a directory")
    c_decompile.add_argument("input_file", help="Input SoundFont file path")
    c_decompile.add_argument("output_directory", nargs="?", help="Output directory to create (default: same name as input file)")

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
            inp = Path(args.input_directory)

            # Determine output file if not specified
            if args.output_file:
                out = Path(args.output_file)
            else:
                # Read info.json to determine format
                info_path = inp / "info.json"
                if not info_path.exists():
                    raise FileNotFoundError(f"info.json not found in {inp}")

                with open(info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)

                # Determine extension based on version
                version = info.get("version", "2.01")
                major_version = int(version.split(".")[0])
                ext = ".sf3" if major_version >= 3 else ".sf2"

                out = inp.with_suffix(ext)

            # Warn if output file exists
            if out.exists():
                response = input(f"Warning: \"{out}\" already exists. Overwrite? (y/n): ")
                if response.lower() != "y":
                    print("Compilation cancelled.")
                    return 0

            compiler = SoundFontCompiler(inp, str(out))
            compiler.compile()

        elif args.command == "decompile":
            sf = Path(args.input_file)

            # Determine output directory if not specified
            if args.output_directory:
                outdir = Path(args.output_directory)
            else:
                # Use the stem of the input file
                outdir = sf.with_suffix("")

            # Warn if output directory exists
            if outdir.exists():
                response = input(f"Warning: \"{outdir}\" already exists. Overwrite? (y/n): ")
                if response.lower() != "y":
                    print("Decompilation cancelled.")
                    return 0

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
