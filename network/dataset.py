import os
import numpy as np
import random
from pathlib import Path

import decimate

LABELS = [
    "idle",
] + list("0123456789")

ROOT = Path("../dataset/samples")

TARGET_DECIMATED_SAMPLES = 30


def load_dataset():
    x = None
    y = np.empty((0,))
    # Collect idle samples for later mixin
    idle_samples = list()
    for root, dirs, files in os.walk(ROOT):
        if "idle" in dirs:
            path = Path(root) / "idle"
            for sample in os.listdir(path):
                idle_samples.append(path / sample)

    # Collect samples
    for root, dirs, files in os.walk(ROOT):
        label = Path(root).parts[-1]
        if label in LABELS:
            for file in files:
                data = decimate.decimated(
                    Path(root) / file,
                    TARGET_DECIMATED_SAMPLES,
                    random.choice(idle_samples),
                )
                data = np.expand_dims(data, axis=0)
                if x is None:
                    x = data
                else:
                    x = np.concatenate((x, data))
                y = np.append(y, LABELS.index(label))

    return x, y.astype(int)
