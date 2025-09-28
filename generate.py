%%writefile generate.py
import torch
from model import TinyTransformer

# ------------------------------
# Load model + vocab
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
stoi = torch.load("stoi.pt")
itos = torch.load("itos.pt")
encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
decode = lambda t: ''.join([itos[int(i)] for i in t])

vocab_size = len(stoi)
block_size = 128
n_layers = 4
n_heads = 4
n_embed = 256
dropout = 0.1

model = TinyTransformer(vocab_size, n_embed, n_layers, n_heads, block_size, dropout).to(device)
model.load_state_dict(torch.load("best_transformer.pt", map_location=device))
model.eval()

# ------------------------------
# Generate text
# ------------------------------
@torch.no_grad()
def sample_prompt(prompt="ROMEO:", tokens=200, temperature=0.9, top_k=50):
    idx = encode(prompt).unsqueeze(0).to(device)
    out = model.generate(idx, max_new_tokens=tokens, temperature=temperature, top_k=top_k)
    return decode(out[0].tolist())

romeo = sample_prompt("ROMEO:", tokens=200)
juliet = sample_prompt("JULIET:", tokens=200)

print("\n=== ROMEO SAMPLE ===\n", romeo[:1000])
print("\n=== JULIET SAMPLE ===\n", juliet[:1000])

with open("sample_ROMEO.txt", "w", encoding="utf-8") as f: f.write(romeo)
with open("sample_JULIET.txt", "w", encoding="utf-8") as f: f.write(juliet)
print("Saved: sample_ROMEO.txt, sample_JULIET.txt")
