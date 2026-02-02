import matplotlib.pyplot as plt

models = ['MLP (Non-Graph)', 'GCN (Graph)']
accuracies = [0.485, 0.814]

plt.bar(models, accuracies)
plt.ylabel('Test Accuracy')
plt.title('Graph vs Non-Graph Learning on Cora Dataset')
plt.ylim(0, 1)
plt.show()
