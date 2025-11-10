import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ComplexRadioDataset(Dataset):
    def __init__(self, data_dir, dtype=np.complex64, max_samples=4096):
        self.data_dir = data_dir
        self.file_pairs = []
        self.dtype = dtype
        self.max_samples = max_samples

        for i in range(5000):
            base = f"{i:06d}"
            complex_path = os.path.join(data_dir, f"{base}.complex")
            text_path = os.path.join(data_dir, f"{base}_text.txt")
            if os.path.exists(complex_path) and os.path.exists(text_path):
                self.file_pairs.append((complex_path, text_path))
        
        print(f"{len(self.file_pairs)} Parov")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        complex_path, text_path = self.file_pairs[idx]

        
        data = np.fromfile(complex_path, dtype=self.dtype)

        
        if self.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
            data = data.view(np.complex64)
        elif self.dtype in [np.complex64, np.float32]:
            max_val = np.abs(data).max() + 1e-8
            data = data / max_val

        if len(data) > self.max_samples:
            data = data[:self.max_samples]
        elif len(data) < self.max_samples:
            padded = np.zeros(self.max_samples, dtype=self.dtype)
            padded[:len(data)] = data
            data = padded

        data_real_imag = np.stack((data.real, data.imag), axis=1).flatten()
        X = torch.tensor(data_real_imag, dtype=torch.float32)

        
        with open(text_path, "r") as f:
            try:
                val = np.float32(f.read().strip())
            except ValueError:
                val = np.float32(0.0)

        if not np.isfinite(val):
            val = np.float32(0.0)

        
        bits = np.unpackbits(np.frombuffer(val.tobytes(), dtype=np.uint8))
        y = torch.tensor(bits.astype(np.float32), dtype=torch.float32)

        return X, y


class ComplexModel32Bit(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),  
            nn.Sigmoid()         
        )

    def forward(self, x):
        return self.net(x)

def train_model(data_dir, epochs=10, batch_size=8, lr=1e-4):
    dataset = ComplexRadioDataset(data_dir)
    sample_X, _ = dataset[0]
    input_dim = len(sample_X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ComplexModel32Bit(input_dim)
    criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "ANN_radio_model_32bit.pth")
    print("DONE")

    return model

if __name__ == "__main__":
    train_model("Training_data3")  
