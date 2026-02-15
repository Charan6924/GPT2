import json
import matplotlib.pyplot as plt
import numpy as np

metrics_file = "log/metrics.jsonl"
train_data = []
val_data = []

with open(metrics_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'loss' in data:
            train_data.append(data)
        if 'validation_loss' in data:
            val_data.append(data)

train_steps = [d['step'] for d in train_data]
train_losses = [d['loss'] for d in train_data]
learning_rates = [d['lr'] for d in train_data]
grad_norms = [d['grad_norm'] for d in train_data]
tokens_per_sec = [d['tokens_per_sec'] for d in train_data]
val_steps = [d['step'] for d in val_data]
val_losses = [d['validation_loss'] for d in val_data]

plt.figure(figsize=(12, 6))
plt.plot(train_steps, train_losses, linewidth=1)
plt.xlabel('Step', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('GPT-2 Training Loss on FineWeb-Edu 10B', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('log/training_loss.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(val_steps, val_losses, 'o-')
plt.xlabel('Step', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Validation Loss', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('log/validation_loss.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# PLOT 3: LEARNING RATE
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(train_steps, learning_rates)
plt.xlabel('Step', fontsize=14)
plt.ylabel('Learning Rate', fontsize=14)
plt.title('Learning Rate Schedule', fontsize=16)
plt.grid(True, alpha=0.3)
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.savefig('log/learning_rate.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(train_steps, grad_norms)
plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Clip threshold')
plt.xlabel('Step', fontsize=14)
plt.ylabel('Gradient Norm', fontsize=14)
plt.title('Gradient Norm', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('log/gradient_norm.png', dpi=300, bbox_inches='tight')
plt.close()


plt.figure(figsize=(12, 6))
plt.plot(train_steps, tokens_per_sec)
plt.xlabel('Step', fontsize=14)
plt.ylabel('Tokens/Second', fontsize=14)
plt.title('Training Throughput', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('log/throughput.png', dpi=300, bbox_inches='tight')
plt.close()
