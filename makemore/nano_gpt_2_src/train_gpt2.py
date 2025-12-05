"""
Implementing GPT-2 from scratch
"""

from dataclasses import dataclass
import torch
import inspect
import torch.nn as nn
import math
import numpy as np
from torch.nn import functional as F
from hellaswag import iterate_examples, render_example


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

def load_tokens(filename):
  npt = np.load(filename)
  ptt = torch.tensor(npt, dtype=torch.long)
  return ptt

class DataLoaderLite:
  def __init__(self, B, T, process_rank, num_processes, split):
    self.B = B
    self.T = T
    self.process_rank = process_rank
    self.num_processes = num_processes
    assert split in {'train', 'val'}


    # ========== Shakespeare ==========
    # # Load and encode data
    # with open('../data/shakespeare.txt') as f:
    #   text = f.read()
    # enc = tiktoken.get_encoding('gpt2')
    # tokens = enc.encode(text)
    # self.tokens = torch.tensor(tokens)
    # print(f"Loaded {len(tokens)} tokens")
    # print(f"1 Epoch contains {self.tokens.size(0) // (self.B * self.T)} batches")

    # ========== FineWeb Shards ==========
    data_root = "data/edu_fineweb10B"
    shards = os.listdir(data_root)
    shards = [s for s in shards if split in s]
    shards = sorted(shards)
    shards = [os.path.join(data_root, s) for s in shards]
    self.shards = shards
    assert len(shards) > 0, f"No shards found for split {split}"
    if master_process:
      print(f"Found {len(shards)} shards for split {split}")
    self.reset()

  def reset(self):
    # Position
    self.current_shard = 0
    self.tokens = load_tokens(self.shards[self.current_shard])
    self.current_position = self.B * self.T * self.process_rank
    
  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position:self.current_position+B*T+1]
    x = (buf[:-1]).view(B, T)
    y = (buf[1:]).view(B, T)

    # Advance the current position
    self.current_position += B * T * self.num_processes

    # Loop around if exceeds all the tokens
    if self.current_position + (B * T * self.num_processes + 1) > self.tokens.size(0):
      self.current_position = self.B * self.T * self.process_rank

    return x, y


# ---------- Helper Functions ----------
def get_most_likely_row(tokens, mask, logits):
  # evaluate the autoregressive loss at all positions
  shift_logits = (logits[..., :-1, :]).contiguous()
  shift_tokens = (tokens[..., 1:]).contiguous()
  flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
  flat_shift_tokens = shift_tokens.view(-1)
  shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
  shift_losses = shift_losses.view(tokens.size(0), -1)
  # now get the average loss just for the completion region (where mask == 1), in each row
  shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
  masked_shift_losses = shift_losses * shift_mask
  # sum and divide by the number of 1s in the mask
  sum_loss = masked_shift_losses.sum(dim=1)
  avg_loss = sum_loss / shift_mask.sum(dim=1)
  # now we have a loss for each of the 4 completions
  # the one with the lowest loss should be the most likely
  pred_norm = avg_loss.argmin().item()
  return pred_norm


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
  master_process = dpp_rank == 0 # master process does logging, checkpointing, etc.
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
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
  print(f"Total desired batch size: {total_batch_size}")
  print(f"=> Calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# IMPROVEMENT: Quantisation
torch.set_float32_matmul_precision('high')

# Create model
# IMPROVEMENT: Make the parameters divisible by 2
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

# IMPROVEMENT: Compile the model for speed up
use_compile = False
if use_compile:
  model = torch.compile(model)
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model


# IMPROVEMENT: Learning rate scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # GPT paper used 375M tokens for warmup: 375e6 / 2**19
max_steps = 10973 # = 10e9 / 2**19

# Total number of tokens: 10e9
# Number of tokens processed per step: 2**19

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


# Initialisation
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
# IMPROVEMENT: Weight decay at initialisation
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# Create a log directory
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w"):
  pass

for step in range(max_steps):
  time0 = time.time()
  last_step = (step == max_steps - 1)

  # Every 250 steps evaluate the validation loss
  if step % 250 == 0:
    model.eval()
    val_loader.reset()
    with torch.no_grad():
      val_loss_accum = 0.0
      val_loss_step = 20
      for _ in range(val_loss_step):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(x, y)
        loss = loss / val_loss_step
        val_loss_accum += loss.detach()
    if ddp:
      dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
    if master_process:
      print(f"Validation loss: {val_loss_accum.item():.4f}")
      with open(log_file, "a") as f:
        f.write(f"Step {step}: Validation loss {val_loss_accum.item():.4f}\n")
      
      # Save model checkpoints 
      if step > 0 and (step % 5000 == 0 or last_step):
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
          'model': raw_model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'config': raw_model.config,
          'step': step,
          'val_loss': val_loss_accum.item(),
        }
        torch.save(checkpoint, checkpoint_path)

  # Every 250 steps evaluate hellaswag
  if (step % 250 == 0 or last_step) and (not use_compile):
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
      # Only process examples where i % ddp_world_size == ddp_rank
      if i % ddp_world_size != ddp_rank:
        continue
      # render the examples into tokens and labels
      _, tokens, mask, label = render_example(example)
      tokens = tokens.to(device)
      mask = mask.to(device)
      with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(tokens)
        pred_norm = get_most_likely_row(tokens, mask, logits)
      num_total += 1
      num_correct_norm += int(pred_norm == label)
    
    if ddp:
      num_total = torch.tensor(num_total, dtype=torch.long, device=device)
      num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
      dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
      dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
      num_total = num_total.item()
      num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
      print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
      with open(log_file, "a") as f:
        f.write(f"{step} hella {acc_norm:.4f}\n")

  # Every 250 steps (except the first one) print samples
  if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
    model.eval()
    num_return_sequences = 4
    max_length = 32
    tokens = enc.encode("Hello, I'm a language model")
    tokens.torch.tensor(tokens, dtype=torch.long)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while xgen.size(1) < max_length:
      with torch.no_grad():
        logits, loss = model(xgen)
        # Take logits at last position
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # Top-k sampling of 50
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
        # Sample from the top-k
        ix = torch.multinomial(topk_probs, num_samples=1, replacement=True, generator=sample_rng)
        xcol = torch.gather(topk_indices, dim=-1, index=ix)
        xgen = torch.cat((xgen, xcol), dim=1)
    for i in range(num_return_sequences):
      tokens = xgen[i, :max_length].tolist()
      decoded = enc.decode(tokens)
      print(f"Rank {ddp_rank} sample {i}: {decoded}")


  # Training
  model.train()
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
    if ddp:
      model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
    loss.backward()

  # Averaging the batched loss across all machines
  if ddp:
    dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)

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
  tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
  tokens_per_sec = tokens_processed / dt

  if master_process:
    print(f"Step {step}: Loss {loss_accum.item():.6f} | lr: {lr:.4f} | Time: {dt:.2f}ms | norm: {norm:.4f} | tok/sec: {tokens_per_sec:.2f}")
    with open(log_file, "a") as f:
      f.write(f"Step {step}: Train loss {loss_accum.item():.6f}\n")

if ddp:
  destroy_process_group()