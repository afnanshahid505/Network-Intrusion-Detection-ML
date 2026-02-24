import matplotlib.pyplot as plt
# Model accuracy values (from your results)
models = ['Baseline Model', 'Proposed Model']
accuracy = [87.10, 88.24]
# Create bar chart
plt.figure(figsize=(6, 4))
plt.bar(models, accuracy)
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison Between Baseline and Proposed Model')
# Add value labels on bars
for i, value in enumerate(accuracy):
    plt.text(i, value + 0.1, f"{value}%", ha='center')
# Save the figure
plt.tight_layout()
plt.savefig("Visualizations/accuracy_comparison.png")
plt.show()