# vision/cnn_bc.py
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

NUM_ACTIONS = 8

class SmallCNN(nn.Module):
    def __init__(self, num_actions=NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):  # x: (B,1,84,84)
        return self.net(x)

def load_dataset(path="../data_collection/artifacts/2025-08-23_23-19-54_human_dataset.npz"):
    data = np.load(path)
    obs = data["obs"]      # (N,84,84,1)
    act = data["act"]      # (N,)

    obs = np.transpose(obs, (0,3,1,2)).astype(np.float32) / 255.0
    return obs, act

def main():
    obs, act = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(obs, act, test_size=0.1, random_state=42, stratify=act)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(NUM_ACTIONS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    def to_tensor(x): return torch.from_numpy(x).to(device)

    for epoch in range(10):
        model.train()
        idxs = np.random.permutation(len(X_train))
        losses = []
        for i in tqdm(range(0, len(idxs), 64)):
            batch = idxs[i:i+64]
            xb = to_tensor(X_train[batch])
            yb = torch.from_numpy(y_train[batch]).long().to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            xv = to_tensor(X_val)
            yv = torch.from_numpy(y_val).long().to(device)
            logits = model(xv)
            val_loss = criterion(logits, yv).item()
            acc = (logits.argmax(dim=1) == yv).float().mean().item()

        print(f"Epoch {epoch+1}: train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f} acc={acc:.3f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/gothic_bc_cnn.pt")
    print("Saved models/gothic_bc_cnn.pt")

if __name__ == "__main__":
    main()
