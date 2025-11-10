import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------- Hyperparameters ----------
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 3000
EVAL_ITERS = 200
LEARNING_RATE = 1e-2
EVAL_INTERVAL = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Params
N_EMBD = 32

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
  idx = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
  x = torch.stack([data[i:i+BLOCK_SIZE] for i in idx])
  y = torch.stack([data[i+1:i+1+BLOCK_SIZE] for i in idx])
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

# ---------- Bigram Model ----------
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    # self.position_embedding_table = nn.Embedding(BLOCK_SIZE, n_embd)
    # self.lm_head = nn.Linear(n_embd, vocab_size) 
  
  def forward(self, idx, targets=None):
    token_emb = self.token_embedding_table(idx) # (B, T, C)
    position_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
    x = token_emb + position_emb
    logits = self.lm_head(x) # (B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ---------- Training Loop ----------
model = BigramLanguageModel()
m = model.to(DEVICE)

optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for iter in range(MAX_ITERS):
  if iter % EVAL_INTERVAL == 0:
    losses = estimate_loss()
    print(f"Step {iter}: Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}")

  X_batch, y_batch = get_batch('train')

  logits, loss = model(X_batch, y_batch)
  optimiser.zero_grad(set_to_none=True)
  loss.backward()
  optimiser.step()

# ---------- Generate From Model ----------
print()
print("Generating text from a model")
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
