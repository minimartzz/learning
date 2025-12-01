"""
Implementing GPT-2 from scratch
"""

from dataclasses import dataclass
import torch
import inspect
import torch.nn as nn
import math
from torch.nn import functional as F

# ---------- Component Definitions ----------
class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0 # Ensure that the number of embedding dimensions can be evenly split among the heads
    # Key, Query, Value projects for ALL heads, but in a batch
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    # Output projections
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1 # setting a flag to scale linear layer standard deviation
    # Regularisation
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    # "Bias" mask - following OpenAI/HF naming
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B, T, C = x.size()
    # B - Batch size
    # T - Sequence length
    # C - Embedding dimensionality
    # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
    # nh: "number of heads", hs: "head size", C: "number of channels" = nh * hs
    # GPT-2 (124M), n_head = 12, hs = 64, nh * hs = C = 768 in the Transformer
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2) # Each token embedding emits a q, k, v vector
    # Below splits into individual heads of attention
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) - B, nh are "batch" dimensions
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) - B, nh are "batch" dimensions
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) - B, nh are "batch" dimensions

    # # Attention (materialised the large (T,T) matrix for all the queries and keys)
    # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    # att = F.softmax(att, dim=-1)
    # y = att @ v # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)

    # IMPROVEMENT: Flash Attention! 🤩
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Compute the output
    y = y.transpose(1, 2).contiguous().view(B, T, C) # reassemble the heads side by side
    y = self.c_proj(y)

    return y

class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
    self.gelu = nn.GELU(approximate="tanh")
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1 # setting a flag to scale linear layer standard deviation
  
  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)
  
  def forward(self, x):
    # Residuals are better to NOT be normalised
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


# ---------- GPT ----------
@dataclass
class GPTConfig:
  block_size: int = 1024   # Max sequence length
  vocab_size: int = 50257  # Number of tokens
  n_layer: int = 12        # Number of layers
  n_head: int = 12         # Number of heads
  n_embd: int = 768        # Embedding dimensions


class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    # Define the overall architecture of the Transformer
    self.transformer = nn.ModuleDict({
      "wte": nn.Embedding(config.vocab_size, config.n_embd),                # Tokens -> Embeddings
      "wpe": nn.Embedding(config.block_size, config.n_embd),                # Positions -> Embeddings
      "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),   # Hidden attention block
      "ln_f": nn.LayerNorm(config.n_embd)                                   # Layer Normalisation
    })
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Prediction head

    # Weight sharing scheme - Final FF layer and initial embedding layer are shared
    self.transformer.wte.weight = self.lm_head.weight

    # Init parameters
    self.apply(self._init_weights)
  
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      std = 0.02
      if hasattr(module, 'NANOGPT_SCALE_INIT'):
        std = (2 * self.config.n_layer) ** -0.5
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
  
  def forward(self, idx, targets=None):
    B, T = idx.size()
    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is the limit on context length"

    # Generate the encodings
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    pos_embd = self.transformer.wpe(pos)
    tok_embd = self.transformer.wte(idx)
    x = pos_embd + tok_embd

    # Pass it through the model
    for layer in self.transformer.h:
      x = layer(x)
    
    # Pass through final layernorm and lm_head
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # (B, T, vocab_size)
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # Flatten the outputs and target

    return logits, loss
  
  @classmethod
  def from_pretrained(cls, model_type):
    """Loads pretrained GPT-2 model weights from huggingface"""
    assert model_type in {'gpt2', 'gpt-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel
    print(f"Loading weights from pretrained gpt: {model_type}")

    # Define different model parameters
    config_args = {
      'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
      'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
      'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
      'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    config_args['vocab_size'] = 50257 # Always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # Always 1024 for GPT model checkpoints

    # Create from-scratch initialised minGPT model
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # Remove bias mask

    # Initialise a hf/transformer model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # Copy while ensuring all the parameters are aligned and match in name and shapes
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    # this means that we have to transpose these weights when we import them
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
      if any(k.endswith(w) for w in transposed):
        # special treatment for the Conv1D weights we need to transpose
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k].t())
      else:
        # vanilla copy over the other parameters
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k])

    return model
  
  def configure_optimizers(self, weight_decay, learning_rate, device):
    # Start with all of the candidate parameters
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # Filter out those that do not require grad
    # create optim gropus. Any parameter that is 2D will be weight decayed, otherwise no.
    # i.e all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
      {"params": decay_params, "weight_decay": weight_decay},
      {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    # Create AdamW optimiser and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device == 'cuda'
    print(f"Using fused AdamW: {use_fused}")
    optimiser = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

    return optimiser
    


# ---------- Data Loader ----------
import tiktoken

class DataLoaderLite:
  def __init__(self, B, T):
    self.B = B
    self.T = T

    # Load and encode data
    with open('../data/shakespeare.txt') as f:
      text = f.read()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)
    self.tokens = torch.tensor(tokens)
    print(f"Loaded {len(tokens)} tokens")
    print(f"1 Epoch contains {self.tokens.size(0) // (self.B * self.T)} batches")

    # Position
    self.current_position = 0
    
  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position:self.current_position+B*T+1]
    x = (buf[:-1]).view(B, T)
    y = (buf[1:]).view(B, T)

    # Advance the current position
    self.current_position += B * T

    # Loop around if exceeds all the tokens
    if self.current_position + (B * T + 1) > self.tokens.size(0):
      self.current_position = 0

    return x, y


# ---------- Training Loop ----------

# ========== Distributed Processing ==========
# set up DDP (distributed data parallel)
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import time

ddp = int(os.environ.get('RANK', -1)) != -1 # Check if it's a ddp run
if ddp:
  assert torch.cuda.is_available(), "DDP requires CUDA"
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  device = f"cuda:{ddp_local_rank}"
  torch.cuda.set_device(device)
  master_process = dpp_rank == 0 # tmaster process does logging, checkpointing, etc.
else:
  # Vanilla non-DDP run
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  master_process = True

  # ========== Auto detect device for single device computation ==========
  # Autodetect device
  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
  # device = "cpu" # Temporary override
  print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
  torch.cuda.manual_seed(1337)

# IMPROVEMENT: Gradient accumulation: simulate any batch size by accumulating mini-batches of gradients
total_batch_size = 2**19 # Original paper batch size is 0.5M so 2**19 is the closest
B = 8    # Micro-batch size
T = 1024  # Length of sequence
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch size: {total_batch_size}")
print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)

# Quantisation
torch.set_float32_matmul_precision('high')

# Get logits
# IMPROVEMENT: Make the parameters divisible by 2
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

# IMPROVEMENT: Compile the model for speed up
model = torch.compile(model)


# Learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
  # Initialisation for warmup
  if it < warmup_steps:
    return max_lr * (it + 1) / warmup_steps

  # If it > lr_decay_iters, returning min learning rate
  if it > max_steps:
    return min_ir
  
  # In between use cosine decay down to min leraning rate
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)


# Training loop
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
# IMPROVEMENT: Weight decay at initialisation
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
  time0 = time.time()
  optimizer.zero_grad()
  loss_accum = 0.0

  # IMPROVEMENT: Simulate any batch size with gradient accumulation
  for _ in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    
    # Mixed precision model - changing logits and loss to BF16 dtype
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
      logits, loss = model(x, y)
    loss = loss / grad_accum_steps # Scale the loss
    loss_accum += loss.detach()
    loss.backward()

  # IMPROVEMENT: Gradient norm clipping
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

  # IMPROVEMENT: Introduce learning rate scheduler - cosine decay
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

  optimizer.step()

  torch.cuda.synchronize() # Uncomment this when timing for GPU
  time1 = time.time()
  dt = (time1 - time0) * 1000
  tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
  tokens_per_sec = tokens_processed / dt

  print(f"Step {step}: Loss {loss_accum.item():.6f} | lr: {lr:.4f} | Time: {dt:.2f}ms | norm: {norm:.4f} | tok/sec: {tokens_per_sec:.2f}")

import sys; sys.exit(0)





# ---------- Run the program ----------

num_return_sequence = 5
max_length = 30

# model = GPT.from_pretrained("gpt2") # Uncomment this to use pretrained GPT-2 weights
model = GPT(GPTConfig())
model.eval()
model.to(device)

# Convert text to tokens
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("The greatest basketball player of all time is,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1)
x = tokens.to('cuda')

# Generate results
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
  with torch.no_grad():
    logits = model(x) # (B, T, vocab_size)
    # Take logits at the last position
    logits = logits[:, -1, :] # (B, vocab_size)
    # Get probabilities
    probs = F.softmax(logits, dim=-1)
    # Perform topk sampling - Takes the top k highest probabilities and their corresponding indices
    # GPT-2 takes top 50 - (B, 50)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    # Sample from the top-k probs
    ix = torch.multinomial(topk_probs, 1) # (B, 1)
    # Get the corresponding column indices
    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
    # Append to sequence
    x = torch.cat((x, xcol), dim=1)

# Print the generated text
for i in range(num_return_sequence):
  tokens = x[i, :max_length].tolist()
  decoded = enc.decode(tokens)
  print(">", decoded)