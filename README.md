# sfutils
A tool to decompile SoundFont (SF2) files into directory structures and recompile them.

## Overview
This tool decompiles SF2 files into a human-readable structure:

```
MySoundFont/
│
├── info.json                   # Metadata (name, copyright, etc.)
│
├── samples/                    # Waveform data (FLAC files + metadata JSON)
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

You can edit the decompiled files and recompile them back into an SF2 file.

## Directory Structure
### Main Tools (src/sfutils/)
- `decompiler.py` - Decompile SF2 files into directories
- `compiler.py` - Compile directories into SF2 files
- `constants.py` - Common constant definitions (Generator IDs, sample types, etc.)

## Usage
### 1. Decompile (SF2 → Directory)
```bash
sfutils decompile <input.sf2> <output_directory>
```

### 2. Compile (Directory → SF2)
```bash
sfutils compile <input_directory> <output.sf2>
```

## Decompiled File Details
### info.json
Contains metadata for the entire SoundFont.

- `name`: SoundFont name
- `version`: SoundFont version
- `creation_date`: Creation date
- `engineer`: Engineer name
- Other metadata fields as available

### samples/ Directory
Each sample is saved in FLAC format.
Each FLAC file has a corresponding .json file with the following information:

- `sample_name`: Sample name
- `sample_type`: Sample type ("mono" or "stereo")
- `start`, `end`: Sample range (relative to the start of the FLAC data)
- `start_loop`, `end_loop`: Loop points (relative to the start of the FLAC data)
- `original_key`: Original pitch (MIDI note number)
- `correction`: Pitch correction value (cents)

The FLAC file itself contains:
- PCM waveform data (16-bit, losslessly compressed)
  - 24-bit is currently not supported in this implementation.
- Sample rate

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
- `bank`: Bank number (0-127, typically 0, drums use 128)
- `preset_number`: Preset number (0-127)
- `zones`: Combine multiple instruments to create one preset

## Limitations
- 24-bit samples (sm24 chunk) are not supported
- sm24 chunks are ignored during decompilation (only 16-bit is used)

## Requirements
- Python 3.12 or higher
- Dependencies: `soundfile`, `numpy` (see pyproject.toml)
