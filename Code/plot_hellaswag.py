import json
import matplotlib.pyplot as plt

results_file = "/mnt/vstor/courses/csds312/cvx166/GPT2/log/hellaswag_results.jsonl"
num_totals = []
accuracies = []
accuracies_norm = []

with open(results_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        num_totals.append(data['num_total'])
        accuracies.append(data['accuracy'])
        accuracies_norm.append(data['accuracy_norm'])

plt.figure(figsize=(10, 6))
plt.plot(num_totals, accuracies, label='Accuracy (sum)', marker='o', markersize=3)
plt.plot(num_totals, accuracies_norm, label='Accuracy (normalized)', marker='s', markersize=3)
plt.axhline(y=0.25, color='r', linestyle='--', label='Random chance (25%)', alpha=0.5)
plt.xlabel('Number of Examples')
plt.ylabel('Accuracy')
plt.title('HellaSwag Evaluation Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('/mnt/vstor/courses/csds312/cvx166/GPT2/log/hellaswag_accuracy.png', dpi=300)
print("Plot saved to /mnt/vstor/courses/csds312/cvx166/GPT2/log/hellaswag_accuracy.png")
plt.show()