import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

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
        print(f"{len(self.file_pairs)} Pairs")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        complex_path, text_path = self.file_pairs[idx]
        data = np.fromfile(complex_path, dtype=self.dtype)

        if self.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
            data = data.view(np.complex64)
        else:
            max_val = np.abs(data).max() + 1e-8
            data = data / max_val

        if len(data) > self.max_samples:
            data = data[:self.max_samples]
        elif len(data) < self.max_samples:
            padded = np.zeros(self.max_samples, dtype=self.dtype)
            padded[:len(data)] = data
            data = padded

        X = torch.tensor(data, dtype=torch.complex64)

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



class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W_real = nn.Linear(in_features, out_features)
        self.W_imag = nn.Linear(in_features, out_features)

    def forward(self, x):
        
        real = self.W_real(x.real) - self.W_imag(x.imag)
        imag = self.W_real(x.imag) + self.W_imag(x.real)
        return torch.complex(real, imag)



class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.complex(torch.relu(x.real), torch.relu(x.imag))



class ComplexModel32Bit(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            ComplexLinear(input_dim, 2048),
            ComplexReLU(),
            ComplexLinear(2048, 1024),
            ComplexReLU(),
            ComplexLinear(1024, 512),
            ComplexReLU(),
            ComplexLinear(512, 512),
            ComplexReLU(),
            ComplexLinear(512, 256),
            ComplexReLU(),
            ComplexLinear(256, 128),
            ComplexReLU(),
            ComplexLinear(128, 32),
        )

    def forward(self, x):
        
        return torch.sigmoid(self.net(x).real)


def train_model(data_dir, epochs=20, batch_size=8, lr=1e-4):
    dataset = ComplexRadioDataset(data_dir)
    sample_X, _ = dataset[0]
    input_dim = sample_X.numel()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ComplexModel32Bit(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            
            X_batch = X_batch.view(X_batch.size(0), -1)
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

    torch.save(model.state_dict(), "COMPLEX2_radio_model_32bit_complex.pth")
    print("DONE")
    return model


if __name__ == "__main__":
    train_model("Training_data3")
