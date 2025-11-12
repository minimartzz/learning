import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------- Hyperparameters ----------
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 5000
EVAL_ITERS = 200
LEARNING_RATE = 1e-3
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


# ---------- Attention Layers ----------
class Head(nn.Module):
  """ Creates a single head of self-attention """
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(N_EMBD, head_size, bias=False)
    self.query = nn.Linear(N_EMBD, head_size, bias=False)
    self.value = nn.Linear(N_EMBD, head_size, bias=False)
    # Buffers are parameters that are not trained by the optimiser
    # https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
    self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
  
  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)

    # Compute attention scores ("affinities")
    wei = q @ k.transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)

    # Perform weighed aggregation of the values
    v = self.value(x)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  """ Multiple hears of self-attention in parallel """
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
  
  def forward(self, x):
    return torch.cat([h(x) for h in self.heads], dim=-1)


# ---------- Feed-forward Layer ----------
class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, n_embd),
      nn.ReLU()
    )
  
  def forward(self, x):
    return self.net(x)


# ---------- Bigram Model ----------
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
    self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
    self.sa_heads = MultiHeadAttention(4, N_EMBD//4)
    self.ffw = FeedForward(N_EMBD)
    self.lm_head = nn.Linear(N_EMBD, vocab_size) 
  
  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_emb = self.token_embedding_table(idx) # (B, T, C)
    position_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T, C)
    x = token_emb + position_emb # (B, T, C)
    x = self.sa_heads(x) # Apply multihead self-attention (B, T, C)
    x = self.ffw(x) # (B, T, C)
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
      # Crop idx to the last block_size token
      idx_cond = idx[:, -BLOCK_SIZE:]
      logits, loss = self(idx_cond)
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
