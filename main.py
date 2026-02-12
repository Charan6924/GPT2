import torch
import torch.nn as nn
import time

# Super simple model
model = nn.Sequential(
    nn.Linear(768, 3072),
    nn.GELU(),
    nn.Linear(3072, 768)
).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Test data
x = torch.randn(16, 1024, 768, device='cuda')
target = torch.randn(16, 1024, 768, device='cuda')

# Warmup
for _ in range(3):
    y = model(x)
    loss = ((y - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Benchmark
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    y = model(x)
    loss = ((y - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()
t1 = time.time()

print(f"Average time per iteration: {(t1-t0)/10*1000:.2f} ms")
print("Expected for 4090: ~50-100ms")