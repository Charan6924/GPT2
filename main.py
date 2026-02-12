import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from train_gpt2 import GPT, GPTConfig
from Dataloaderlite import Dataloaderlite
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

model = GPT(GPTConfig()).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
train_loader = Dataloaderlite(16, 1024)

# Warmup
for _ in range(3):
    x, y = train_loader.next_batch()
    x, y = x.to('cuda'), y.to('cuda')
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    
    # Explicit cleanup
    del x, y, logits, loss
    torch.cuda.empty_cache()

print("\nTimed runs:")
for i in range(10):
    x, y = train_loader.next_batch()
    x, y = x.to('cuda'), y.to('cuda')
    
    torch.cuda.synchronize()
    t0 = time.time()
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    
    dt = (time.time()-t0)*1000
    print(f"Step {i}: {dt:.2f}ms, loss: {loss.item():.4f}")
    
    # Cleanup after each step