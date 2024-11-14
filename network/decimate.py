import pandas as pd
import numpy as np
import io


def get_samples(path):
    with open(path, "r") as f:
        buffer = f.read().replace(",", "\t")
    buffer = io.StringIO(buffer)

    msft = pd.read_csv(buffer, sep="\t", names=["time", "x", "y", "z"])

    # Normalize for time offset
    msft["time"] -= msft["time"][0]

    out = dict()

    for axis in [
        "time",
    ] + list("xyz"):
        out[axis] = np.array(msft[axis], dtype=float)

    return out


def decimated(path, target_samples, path_unknown, max_time_ms=3000):
    data_sample = get_samples(path)

    # pad with samples of the category unknown
    data_pad = get_samples(path_unknown)

    data_pad["time"] += data_sample["time"][-1] + max_time_ms / target_samples

    msft = dict()
    for axis, data in data_sample.items():
        msft[axis] = np.concatenate((data_sample[axis], data_pad[axis]))

    time = np.linspace(0, max_time_ms, target_samples)
    x, y, z = [np.interp(time, msft["time"], msft[axis]) for axis in "xyz"]

    data = np.array([x, y, z]).T

    return data
