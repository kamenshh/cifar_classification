from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt



class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.drop1 = torch.nn.Dropout(0.25)

        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.drop2 = torch.nn.Dropout(0.25)

        self.fc1 = torch.nn.Linear(128 * 8 * 8, 256)
        self.drop3 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    csv = pd.read_csv(project_dir / "cifar10.csv")
    y = np.array(csv.iloc[:, 0])
    X = np.array(csv.iloc[:, 1:], dtype=np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


    X = np.array(csv.iloc[:, 1:], dtype=np.float32) / 255.0
    X = X.reshape(-1, 3, 32, 32)
    X = (X - mean) / std
    

    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=42
    )

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_te = torch.tensor(y_test, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_tr, y_tr)
    test_dataset = TensorDataset(X_te, y_te)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    for epoch in range(30):
        model.train()
        total_loss = 0
        for batch_x, batch_y in tqdm(train_loader):
            mask = torch.rand(batch_x.size(0), device=device) < 0.5
            batch_x[mask] = torch.flip(batch_x[mask], dims=[3])
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = F.cross_entropy(pred, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                pred = model(batch_x)
                correct += (pred.argmax(dim=1) == batch_y).sum().item()
                total += batch_y.size(0)

        print(f"Epoch {epoch + 1}, loss = {total_loss:.4f}, accuracy = {correct / total:.4f}")

    torch.save(model.state_dict(), project_dir / "cnn_cifar10.pth")
    print("Model saved!")
