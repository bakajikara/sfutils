# sfutils
A tool to decompile SoundFont (SF2/SF3) files into directory structures and recompile them.

## Overview
This tool decompiles SF2/SF3 files into a human-readable structure:

```
MySoundFont/
│
├── info.json                   # Metadata (name, copyright, version, etc.)
│
├── samples/                    # Waveform data (FLAC/OGG files + metadata JSON)
│   ├── Piano FF A0.flac
│   ├── Piano FF A0.json        # Loop points, sample type, etc.
│   ├── Piano MF A0.flac
│   ├── Piano MF A0.json
│   └── ...
│
├── instruments/                # Instrument definitions (JSON)
│   ├── Accordion.json
│   ├── Acoustic Bass.json
│   └── ...
│
└── presets/                    # Preset definitions (JSON)
    ├── 000-000_Grand Piano.json
    ├── 000-001_Bright Grand Piano.json
    └── ...
```

You can edit the decompiled files and recompile them back into an SF2 or SF3 file.

## Directory Structure
### Main Tools (src/sfutils/)
- `decompiler.py` - Decompile SF2/SF3 files into directories
- `compiler.py` - Compile directories into SF2/SF3 files
- `parser.py` - Parse SoundFont file structures
- `constants.py` - Common constant definitions (Generator IDs, sample types, etc.)

## Usage
### 1. Decompile (SF2/SF3 → Directory)
```bash
sfutils decompile <input_file> [output_directory] [options]
```

**Options:**
- `-f, --force`: Force overwrite without confirmation

If the output directory is not specified, it defaults to the input filename without extension.
If the output directory already exists and `--force` is not specified, you will be prompted to confirm overwrite.

### 2. Compile (Directory → SF2/SF3)
```bash
sfutils compile <input_directory> [output_file] [options]
```

**Options:**
- `-q, --quality QUALITY`: Ogg Vorbis quality for SF3 (0.0-1.0, default: 0.8)
  - Higher values = better quality but larger file size
  - Only affects SF3 files; ignored for SF2
- `-f, --force`: Force overwrite without confirmation

If the output file is not specified, the output filename is automatically determined from the input directory name. The file extension (`.sf2` or `.sf3`) is determined by the `version` field in `info.json`:
- Version 2.x → `.sf2`
- Version 3.x → `.sf3`

If the output file already exists and `--force` is not specified, you will be prompted to confirm overwrite.

## Decompiled File Details
### info.json
Contains metadata for the entire SoundFont.

- `version`: SoundFont version (e.g., "2.01" for SF2, "3.01" for SF3)
- `sound_engine`: Sound engine name
- `bank_name`: SoundFont name
- `creation_date`: Creation date
- `engineer`: Engineer/author name
- `product`: Product information
- `copyright`: Copyright information
- `comment`: Additional comments
- Other metadata fields as available

### samples/ Directory
Each sample is saved in FLAC format (for SF2) or OGG Vorbis format (for SF3).
For SF2 files, stereo-linked samples are automatically detected and combined into a single stereo FLAC file with two channels.
For SF3 files, stereo samples are saved as separate mono OGG files (e.g., `sample_L.ogg` and `sample_R.ogg`).
Each audio file has a corresponding .json file with the following information:

- `sample_name`: Sample name
- `sample_type`: "mono" or "stereo"
- `start_loop`: Loop start position (relative to sample start)
- `end_loop`: Loop end position (relative to sample start)
- `original_key`: Original pitch (MIDI note number, 0-127)
- `correction`: Pitch correction value (cents, -99 to 99)

**Audio Format:**
- **SF2 decompilation**: Samples are saved as 16-bit or 24-bit FLAC files (losslessly compressed PCM)
  - 24-bit support is available via the sm24 chunk
- **SF3 decompilation**: Samples are saved as OGG Vorbis files (preserving original compression)
  - Stereo samples are saved as separate mono files

### instruments/ Directory
Each instrument is saved as a separate JSON file.

**Zones**: Instruments consist of multiple zones.
- **Global Zone**: Common settings applied to all local zones (optional; first zone without a sample)
- **Local Zones**: Sample and settings for specific key/velocity ranges

**Generators**: Sound parameters
- `keyRange`: MIDI note range for this zone (e.g., "36-72")
- `velRange`: Velocity range (e.g., "0-127")
- `sample`: Sample name to use
- `initialFilterFc`: Filter frequency
- `attackVolEnv`: Envelope attack time
- Many other parameters (see SF2 specification)

### presets/ Directory
Each preset (MIDI program) is saved as a separate JSON file.
Filename format: `{bank:03d}-{preset:03d}_{name}.json`

**Preset**: Sounds selected by MIDI program change.
- `bank`: Bank number (0-127 for melodic instruments, 128 for percussion)
- `preset_number`: Preset number (0-127)
- `zones`: Combine multiple instruments to create one preset

## File Format Support
### SF2 (SoundFont 2)
- **Sample Format**: 16-bit or 24-bit PCM
- **Decompilation**: Samples are saved as FLAC files (lossless compression)
  - 16-bit and 24-bit samples are both supported
- **Compilation**: Audio files supported by `soundfile` (FLAC, WAV, etc.) are converted to PCM
  - The compiler automatically detects bit depth and handles 16-bit/24-bit appropriately
- **Stereo Handling**:
  - Decompilation: Stereo-linked samples are combined into a single stereo FLAC file
  - Compilation: Stereo FLAC files are automatically split into left/right channels
- **Version**: Defaults to 2.01

### SF3 (SoundFont 3)
- **Sample Format**: Ogg Vorbis (lossy compression)
- **Decompilation**: Samples are saved as OGG files (preserving original encoding)
- **Compilation**:
  - OGG files are used directly (no re-encoding required)
  - Other audio formats supported by `soundfile` can be used and will be encoded to Ogg Vorbis
  - Encoding to Ogg Vorbis requires `ffmpeg-python` and FFmpeg
  - Quality can be controlled with `--quality` option (0.0-1.0, default: 0.8)
    - 0.0 = lowest quality (smallest file)
    - 1.0 = highest quality (largest file)
- **Stereo Handling**:
  - Decompilation: Stereo samples are saved as separate mono OGG files (`_L.ogg` and `_R.ogg`)
  - Compilation: Mono samples are used as-is
- **Version**: Defaults to 3.01

The format is automatically detected during decompilation and determined by the output file extension during compilation.

## Limitations
- SF3 stereo samples are decompiled as separate mono files (not combined into stereo)
- SF3 compilation from non-OGG formats requires `ffmpeg-python` library and FFmpeg to be installed
- Only OGG files can be used for SF3 compilation without FFmpeg

## Requirements
- Python 3.12 or higher
- Dependencies:
  - `soundfile`: For audio file I/O (required for all operations)
  - `numpy`: For numerical operations (required for all operations)
  - `ffmpeg-python`: For encoding non-OGG files to Ogg Vorbis (only required for SF3 compilation from non-OGG sources)

**Note:** FFmpeg must be installed on your system to use `ffmpeg-python` for Ogg Vorbis encoding.

See `pyproject.toml` for the complete dependency list.
