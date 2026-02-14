import json
import matplotlib.pyplot as plt

# Load metrics
metrics = []
with open('log/metrics.jsonl', 'r') as f:
    for line in f:
        metrics.append(json.loads(line))

# Extract data
steps = [m['step'] for m in metrics]
losses = [m['loss'] for m in metrics]
lrs = [m['lr'] for m in metrics]
grad_norms = [m['grad_norm'] for m in metrics]
tokens_per_sec = [m['tokens_per_sec'] for m in metrics]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss curve
axes[0, 0].plot(steps, losses, linewidth=1.5)
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].grid(True, alpha=0.3)

# Learning rate schedule
axes[0, 1].plot(steps, lrs, linewidth=1.5, color='orange')
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('Learning Rate')
axes[0, 1].set_title('Learning Rate Schedule')
axes[0, 1].grid(True, alpha=0.3)

# Gradient norm
axes[1, 0].plot(steps, grad_norms, linewidth=1.5, color='green')
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('Gradient Norm')
axes[1, 0].set_title('Gradient Norm')
axes[1, 0].grid(True, alpha=0.3)

# Throughput
axes[1, 1].plot(steps, tokens_per_sec, linewidth=1.5, color='red')
axes[1, 1].set_xlabel('Step')
axes[1, 1].set_ylabel('Tokens/sec')
axes[1, 1].set_title('Training Throughput')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
print("Saved training_metrics.png")

# Optional: Just loss curve
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, linewidth=2)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('GPT-2 Training Loss on FineWeb-Edu 10B', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
print("Saved loss_curve.png")