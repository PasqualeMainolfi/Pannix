import numpy as np
import numpy.typing as npt
import wave
import os


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
    y = np.nan_to_num(out)
    return y


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

def export_multitrack(frames: list[list], name: str, sample_rate: int = 44100, sampwidth: int = 2):

    """
    frames: list[list], non bytes list of frames
    name: str, folder name
    sample_rate: int, sample rate
    sampwidth: int, sampwidth
    """
    
    f = np.array(frames, dtype=object)
    y = f[0].copy()

    for i in range(1, len(frames[0])):
        y = np.concatenate((y, f[i]))
    
    y = y.T

    cd = os.getcwd()
    folder = os.path.join(cd, name)
    
    try:
        os.mkdir(folder)
    except OSError as e:
        print(e)

    nchnls = len(y)

    for j in range(nchnls):
        path = os.path.join(folder, f"chn{j+1}.wav")
        out = wave.open(path, "wb")
        out.setnchannels(1)
        out.setframerate(sample_rate)
        out.setsampwidth(sampwidth)
        out.writeframes(y[j].tobytes())
        out.close()



