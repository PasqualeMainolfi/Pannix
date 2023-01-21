import numpy as np
import numpy.typing as npt


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
    out = np.zeros((len(data)//nchnls, nchnls), dtype=format)
    for i in range(nchnls):
        out[:, i] = data[i::nchnls]
    return out