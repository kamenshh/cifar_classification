from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from education import CNN


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    csv = pd.read_csv(project_dir / "cifar10.csv")
    y = np.array(csv.iloc[:, 0])
    X = np.array(csv.iloc[:, 1:], dtype=np.float32) / 255.0

    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)

    X = X.reshape(-1, 3, 32, 32)
    X = (X - mean) / std
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=42
    )

    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_te = torch.tensor(y_test, dtype=torch.long).to(device)

    model = CNN().to(device)
    model.load_state_dict(torch.load(project_dir / "cnn_cifar10.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(X_te)
        y_pred = pred.argmax(dim=1).cpu().numpy()
        y_true = y_te.cpu().numpy()

    correct = (y_pred == y_true).sum()
    accuracy = correct / len(y_true)
    print("Accuracy:", round(accuracy, 4))
    print()
    print(classification_report(y_true, y_pred, zero_division=0))
