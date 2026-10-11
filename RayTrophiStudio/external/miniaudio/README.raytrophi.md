# Vendored miniaudio

Version: 0.11.23. Official source and license downloaded unmodified from
https://github.com/mackron/miniaudio/tree/0.11.23.

Implementation is compiled exactly once in source/src/Scene/AudioService.cpp.
MA_NO_ENCODING and MA_NO_GENERATION omit unused encoder/generator features.
WAV, FLAC and MP3 decoders and default runtime-linked device backends remain enabled.

Retain LICENSE with source/binary distributions. No Opus decoder or third-party
audio content is included. Opus/streaming and EXE packaging are separate future work.
