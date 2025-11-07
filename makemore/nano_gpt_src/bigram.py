import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------- Hyperparameters ----------
BATCH_SIZE = 32
EVAL_ITERS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(1337)

# ---------- Load Data ----------
with open("../data/shakespeare.txt", "r") as f:
  text = f.read()
print(f"Length of dataset in chraceters: {len(text)}")

# Define unique dataset parameters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mappers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x] # Convert strings to index
decode = lambda x: ''.join([itos[c] for c in x]) # Convert index to strings

# Convert text into indices and split data
data = torch.tensor(encode(text), dtype=torch.long)
sp = 0.9
idx = int(sp * len(data))
train_data = data[:idx]
val_data = data[idx:]

# ---------- Batch Data Function ----------
def get_batch(split):
  data = train_data if split == 'train' else val_data
  idx = torch.randint(len(data) - block_size, (BATCH_SIZE,))
  x = torch.stack([data[i:i+block_size] for i in idx])
  y = torch.stack([data[i+1:i+1+block_size] for i in idx])
  x, y = x.to(DEVICE), y.to(DEVICE)

  return x, y

# ---------- Evalute Model Function ----------
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
      X, y = get_batch(split)
      logits, loss = model(X, y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out