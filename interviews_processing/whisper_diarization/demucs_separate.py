"""Run `demucs.separate` with WAV stems written by soundfile.

Recent torchaudio versions delegate `torchaudio.save` to torchcodec, which needs a
shared FFmpeg build (DLLs) on Windows. Without it demucs crashes when saving the
stems and diarize.py silently falls back to the unseparated audio.

Usage is identical to `python -m demucs.separate ...`.
"""
import soundfile as sf

import demucs.separate
from demucs.audio import prevent_clip

_original_save_audio = demucs.separate.save_audio


def _save_audio(wav, path, samplerate, clip="rescale", bits_per_sample=16, as_float=False, **kwargs):
    if not str(path).lower().endswith(".wav"):
        return _original_save_audio(
            wav, path, samplerate, clip=clip, bits_per_sample=bits_per_sample, as_float=as_float, **kwargs
        )
    wav = prevent_clip(wav, mode=clip)
    subtype = "FLOAT" if as_float else f"PCM_{bits_per_sample}"
    # demucs tensors are (channels, samples); soundfile expects (samples, channels)
    sf.write(str(path), wav.T.cpu().numpy(), samplerate, subtype=subtype)


demucs.separate.save_audio = _save_audio

if __name__ == "__main__":
    demucs.separate.main()
