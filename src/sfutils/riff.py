# Copyright (c) 2025 bakajikara
#
# This file is licensed under the MIT License (MIT).
# See the LICENSE file in the project root for the full license text.

"""
RIFF (Resource Interchange File Format) utility functions.
Provides functions to read and create RIFF chunks, including LIST chunks.
"""

import struct
from typing import BinaryIO


def read_chunk_header(f: BinaryIO) -> tuple[bytes, int]:
    """
    Reads a RIFF chunk header (ID and size) from a file.

    Args:
        f: The file object to read from.

    Returns:
        A tuple containing the chunk ID (bytes) and chunk size (int).
    """
    chunk_id = f.read(4)
    if len(chunk_id) < 4:
        raise EOFError("Unexpected end of file while reading chunk ID.")

    chunk_size_bytes = f.read(4)
    if len(chunk_size_bytes) < 4:
        raise EOFError("Unexpected end of file while reading chunk size.")

    chunk_size = struct.unpack("<I", chunk_size_bytes)[0]
    return chunk_id, chunk_size


def make_chunk(chunk_id: bytes, data: bytes) -> bytes:
    """
    Creates a RIFF chunk with the given ID and data.
    Automatically adds padding if the data size is odd.

    Args:
        chunk_id: The 4-byte chunk ID.
        data: The chunk's data.

    Returns:
        The created chunk as a bytes object.
    """
    if len(chunk_id) != 4:
        raise ValueError("Chunk ID must be 4 bytes long.")

    size = len(data)
    packed_data = chunk_id + struct.pack("<I", size) + data

    # Add padding if size is odd
    if size % 2:
        packed_data += b"\x00"

    return packed_data


def make_list_chunk(list_type: bytes, data: bytes) -> bytes:
    """
    Creates a LIST chunk with the given list type and data.

    Args:
        list_type: The 4-byte list type ID (e.g., b"INFO", b"pdta").
        data: The internal data to be wrapped in the LIST chunk.

    Returns:
        The created LIST chunk as a bytes object.
    """
    if len(list_type) != 4:
        raise ValueError("List type ID must be 4 bytes long.")

    # Internal data starts with the list type ID
    list_data = list_type + data
    return make_chunk(b"LIST", list_data)


def make_zstr(text: str, encoding: str = "ascii") -> bytes:
    """
    Creates a zero-terminated string compliant with the RIFF specification.
    Adjusts the total byte count to be even by adding one or two null terminators.

    Args:
        text: The string to convert.
        encoding: The encoding (default: "ascii").

    Returns:
        The zero-terminated string as a bytes object.
    """
    encoded = text.encode(encoding)

    # If string length is odd, add one terminator (total even)
    # If string length is even, add two terminators (total even)
    if len(encoded) % 2 == 1:
        return encoded + b"\x00"
    else:
        return encoded + b"\x00\x00"
