from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "cifar10.csv"


if __name__ == "__main__":
    csv = pd.read_csv(CSV_PATH)
    y = np.array(csv.iloc[:, 0])
    X = np.array(csv.iloc[:, 1:], dtype=np.float32) / 255.0


    counter = Counter(y)
    print("Images:", len(X))
    print("Classes:", len(counter))
    print("Min class count:", min(counter.values()))
    print("Max class count:", max(counter.values()))

    images = X.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    print("Mean by channels:", (images.mean(axis=(0, 1, 2))).round(4))
    print("Std by channels:", (images.std(axis=(0, 1, 2))).round(4))

    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(f"class {y[i]}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
