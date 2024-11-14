import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DIRECTORY = "samples"

for root, dirs, files in os.walk(DIRECTORY):
    for file in sorted(files):
        path = f"{root}/{file}"
        print(path)
        msft = pd.read_csv(path, sep="\t", names=["time", "x", "y", "z"])
        msft["time"] -= msft["time"][0]
        print(msft)
        msft.plot(0, marker="o", linestyle="dashed")
        plt.show()

        time = np.linspace(0, 3000, 60)

        x, y, z = [np.interp(time, msft["time"], msft[axis]) for axis in "xyz"]

        plt.close()

        data = np.array([time, x, y, z]).T

        print(data)

        interpolated = pd.DataFrame(data)

        interpolated.plot(0, marker="o", linestyle="dashed")
        plt.show()

        plt.close()
