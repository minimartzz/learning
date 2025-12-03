"""
FineWeb-Edu dataset for GPT-2 Pretraining

Run using:
$ python edu_fineweb10B.py

Saves shares to the local directory
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ------------------------- Variables ------------------------- 
LOCAL_DIR = "data/edu_fineweb10B"
REMOTE_NAME = "sample-10BT"
SHARD_SIZE = int(1e8) # 100M tokens per shard, total of 100 shards

# Create the cache as local directory if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), LOCAL_DIR)
if not os.path.exists(DATA_CACHE_DIR):
  os.makedirs(DATA_CACHE_DIR)


# ------------------------- Process ------------------------- 
# Download dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=REMOTE_NAME, split="train")

# Init tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens('<|endoftext|>') # EOT token
def tokenizer(doc):
  # Tokenize a single document adn returns a numpy array of uint16 tokens
  tokens = [eot]
  tokens.extend(enc.encode_ordinary(doc['text']))
  tokens_np = np.array(tokens)

  assert (0 <= tokens_np).all() and (tokens_np <= 2**16).all(), "token dictionary too large for uint16"
  tokens_np_uint16 = tokens_np.astype(np.uint16)
  return tokens_np_uint16

def write_datafile(filename, tokens_np):
  np.save(filename, tokens_np)

# Tokenise all documents and write output shards, each of SHARD_SIZE tokens
nprocs = max(1, os.cpu_count() // 2) # Maximise the number of CPUs performing the task
with mp.Pool(nprocs) as pool:
  shard_index = 0

  # Preallocate buffer to hold current shard
  all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
  token_count = 0
  progress_bar = None

  for tokens in pool.imap(tokenize, fw, chunksize=16):
    # check if enough space in current shard for new tokens
    if token_count + len(tokens) < SHARD_SIZE:
      all_tokens_np[token_count:token_count+len(tokens)] = tokens
      token_count += len(tokens)
      # update progress bar
      if progress_bar is None:
        progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}")
      progress_bar.update(len(tokens))
    else:
      # write the current shard and start a new one
      split = 'val' if shard_index == 0 else 'train'
      filename = ox.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy")
      # split the document into whatever fits in this shard; remainder goes to next one
      remainder = SHARD_SIZE - token_count
      progress_bar.update(remainder)
      all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]

      write_datafile(filename, all_tokens_np)
      shard_index += 1
      progress_bar = None
      # populate the next shard with the leftovers of the current doc
      all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
      token_count = len(tokens)-remainder
  
  # write any remaining tokens as the last shard
  if token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])