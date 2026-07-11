import pickle
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARCHIVE_PATH = ROOT / "cifar-10-python.tar.gz"
CSV_PATH = ROOT / "cifar10.csv"


def load_batch(file):
    data = pickle.load(file, encoding="latin1")
    X = data["data"]
    y = data["labels"]
    return X, y


if __name__ == "__main__":
    all_X = []
    all_y = []

    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        for member in tar.getmembers():
            if "data_batch_" not in member.name:
                continue

            file = tar.extractfile(member)
            X, y = load_batch(file)
            all_X.append(X)
            all_y.extend(y)

    X = np.vstack(all_X)
    y = np.array(all_y)

    rows = np.column_stack([y, X])
    columns = ["label"] + [f"pixel_{i}" for i in range(32 * 32 * 3)]
    csv = pd.DataFrame(rows, columns=columns)
    csv.to_csv(CSV_PATH, index=False)

    print("Dataset saved to", CSV_PATH)
