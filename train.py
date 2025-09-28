%%writefile train.py
import torch, requests, os
from torch.utils.data import Dataset, DataLoader
from model import TinyTransformer
import torch.nn.functional as F

# ------------------------------
# Config
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
block_size = 128
n_layers = 4
n_heads = 4
n_embed = 256
dropout = 0.1
lr = 3e-4
epochs = 5

# ------------------------------
# Load dataset
# ------------------------------
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text_path = "tinyshakespeare.txt"
if not os.path.exists(text_path):
    with open(text_path, "wb") as f:
        f.write(requests.get(url).content)

with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
decode = lambda t: ''.join([itos[int(i)] for i in t])

data = encode(text)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

class CharDataset(Dataset):
    def __init__(self, data_tensor, block_size):
        self.data = data_tensor
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

train_loader = DataLoader(CharDataset(train_data, block_size), batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate)
val_loader   = DataLoader(CharDataset(val_data, block_size), batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate)

# ------------------------------
# Model + Optimizer
# ------------------------------
model = TinyTransformer(vocab_size, n_embed, n_layers, n_heads, block_size, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ------------------------------
# Eval helper
# ------------------------------
def evaluate(loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses)/len(losses)

# ------------------------------
# Training loop
# ------------------------------
best_val = float('inf')
for epoch in range(1, epochs+1):
    for i, (x, y) in enumerate(train_loader, start=1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            train_loss = loss.item()
            val_loss = evaluate(val_loader)
            print(f"Epoch {epoch} | Step {i}/{len(train_loader)} | Train {train_loss:.4f} | Val {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), "best_transformer.pt")
                torch.save(stoi, "stoi.pt")
                torch.save(itos, "itos.pt")

print("Training finished. Model saved to best_transformer.pt")
