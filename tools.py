import numpy as np
import numpy.typing as npt
import wave


def to_nchnls(input_sig: bytes, nchnls: int, format: npt.DTypeLike) -> npt.ArrayLike:

    """
    convert in n channels
    inputs_sig: bytes
    nchnls: int
    format: dtype

    return: npt.ArrayLike
    """

    if nchnls == 1: return input_sig
    data = np.frombuffer(input_sig, dtype=format)
    out = np.zeros((len(data), nchnls), dtype=format)
    for i in range(nchnls):
        out[:, i] = data[::]
    return out



def save_audio_file(path: str, frames: list[list], sample_rate: int, nchnls: int, sampwidth: int) -> None:
    
    """
    path: str, path to save
    frames: list[list], audio file in bytes format
    sample_rate: int, sample rate
    nchnls: int, number of channels
    sampwidth: int, sampwidth
    """

    file = wave.open(path, "wb")
    file.setnchannels(nchnls)
    file.setframerate(sample_rate)
    file.setsampwidth(sampwidth)
    file.writeframes(b''.join(frames))
    file.close()
