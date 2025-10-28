#!/usr/bin/env python3
"""
SF2 Constants - SoundFont2ファイルで使用する共通定数定義

コンパイラとデコンパイラの両方で使用される定数を定義します。
"""

# ジェネレータ名からIDへのマッピング
# SF2 2.04 spec section 8.1.2 - Generator Enumerators
GENERATOR_IDS = {
    "startAddrsOffset": 0,
    "endAddrsOffset": 1,
    "startloopAddrsOffset": 2,
    "endloopAddrsOffset": 3,
    "startAddrsCoarseOffset": 4,
    "modLfoToPitch": 5,
    "vibLfoToPitch": 6,
    "modEnvToPitch": 7,
    "initialFilterFc": 8,
    "initialFilterQ": 9,
    "modLfoToFilterFc": 10,
    "modEnvToFilterFc": 11,
    "endAddrsCoarseOffset": 12,
    "modLfoToVolume": 13,
    "unused1": 14,
    "chorusEffectsSend": 15,
    "reverbEffectsSend": 16,
    "pan": 17,
    "unused2": 18,
    "unused3": 19,
    "unused4": 20,
    "delayModLFO": 21,
    "freqModLFO": 22,
    "delayVibLFO": 23,
    "freqVibLFO": 24,
    "delayModEnv": 25,
    "attackModEnv": 26,
    "holdModEnv": 27,
    "decayModEnv": 28,
    "sustainModEnv": 29,
    "releaseModEnv": 30,
    "keynumToModEnvHold": 31,
    "keynumToModEnvDecay": 32,
    "delayVolEnv": 33,
    "attackVolEnv": 34,
    "holdVolEnv": 35,
    "decayVolEnv": 36,
    "sustainVolEnv": 37,
    "releaseVolEnv": 38,
    "keynumToVolEnvHold": 39,
    "keynumToVolEnvDecay": 40,
    "instrument": 41,
    "reserved1": 42,
    "keyRange": 43,
    "velRange": 44,
    "startloopAddrsCoarseOffset": 45,
    "keynum": 46,
    "velocity": 47,
    "initialAttenuation": 48,
    "reserved2": 49,
    "endloopAddrsCoarseOffset": 50,
    "coarseTune": 51,
    "fineTune": 52,
    "sampleID": 53,
    "sampleModes": 54,
    "reserved3": 55,
    "scaleTuning": 56,
    "exclusiveClass": 57,
    "overridingRootKey": 58,
    "unused5": 59,
    "endOper": 60
}

# ジェネレータIDから名前へのマッピング（デコンパイラ用）
GENERATOR_NAMES = {id_: name for name, id_ in GENERATOR_IDS.items()}

# サンプルタイプ
SAMPLE_TYPES = {
    1: "monoSample",
    2: "rightSample",
    4: "leftSample",
    8: "linkedSample"
}

# サンプルタイプの逆引き（名前からIDへ）
SAMPLE_TYPE_IDS = {name: id_ for id_, name in SAMPLE_TYPES.items()}
