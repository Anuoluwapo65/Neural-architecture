import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class RNNConfig:
    device: torch.device = torch.device("cpu")
    input_size: int = 1
    hidden_size: int = 32
    batch_size: int = 32
    learning_rate: float = 1e-3
    dropout: float = 0.1
    epochs: int = 200
    num_samples: int = 10000
    seq_length: int = 16
    train_fraction: float = 0.8


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
config = RNNConfig()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(config.device)


t = torch.linspace(0, 100, config.num_samples + config.seq_length, device=config.device)
signal = torch.sin(t)

x_tensor = torch.stack([signal[i:i + config.seq_length] for i in range(config.num_samples)])
y_tensor = signal[config.seq_length:config.num_samples + config.seq_length]

train_size = int(config.train_fraction * config.num_samples)
x_train, y_train = x_tensor[:train_size], y_tensor[:train_size]
x_val, y_val = x_tensor[train_size:], y_tensor[train_size:]


class SequenceDataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs.float()
        self.targets = targets.float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.inputs[idx].unsqueeze(-1)
        y = self.targets[idx]
        return x, y


train_dataset = SequenceDataset(x_train, y_train)
val_dataset = SequenceDataset(x_val, y_val)

train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        hidden, _ = self.rnn(x)
        hidden = self.dropout(hidden[:, -1, :])
        return self.output(hidden).squeeze(-1)


model = SimpleRNN(
    input_size=config.input_size,
    hidden_size=config.hidden_size,
    dropout=config.dropout,
).to(config.device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

train_losses = []
val_losses = []

for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(config.device)
        batch_y = batch_y.to(config.device)

        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)
            predictions = model(batch_x)
            val_loss += criterion(predictions, batch_y).item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    if (epoch + 1) % 25 == 0 or epoch == config.epochs - 1:
        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f}"
        )

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="train_loss")
plt.plot(val_losses, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RNN Training Curve")
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    sample_x = x_val[:5].to(config.device)
    sample_y = y_val[:5].to(config.device)
    predictions = model(sample_x.unsqueeze(-1))

    print("Actual values:", sample_y.cpu().numpy())
    print("Predicted values:", predictions.cpu().numpy())









