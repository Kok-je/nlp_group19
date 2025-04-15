import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("results/Trained/loss.csv",header=None)
data.columns = ["loss","grad_norm","learning_rate","epoch"]
data.loss = data.loss.apply(lambda x: x[9:])
data.grad_norm = data.grad_norm.apply(lambda x: x[13:])
data.learning_rate = data.learning_rate.apply(lambda x: x[18:])
data.epoch = data.epoch.apply(lambda x: x[9:-1])

data['epoch'] = data['epoch'].astype(float)
data['loss'] = data['loss'].astype(float)
data['grad_norm'] = data['grad_norm'].astype(float)
data['learning_rate'] = data['learning_rate'].astype(float)

# Create figure with primary y-axis
fig, ax1 = plt.subplots(figsize=(10, 6))
#plt.style.use("seaborn-v0_8-white")
#ax1.set_facecolor((1, 1, 1, 0.8))

# Plot loss
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
sns.lineplot(x='epoch', y='loss', data=data, ax=ax1, color=color, label='Loss',legend=False)
ax1.tick_params(axis='y', labelcolor=color)

# Create secondary y-axis for grad_norm
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Gradient Norm', color=color)
sns.lineplot(x='epoch', y='grad_norm', data=data, ax=ax2, color=color, label='Gradient Norm',legend=False)
ax2.tick_params(axis='y', labelcolor=color)

# # Create third y-axis for learning_rate
# ax3 = ax1.twinx()
# ax3.spines["right"].set_position(("axes", 1.1))  # Offset the right spine
# color = 'tab:green'
# ax3.set_ylabel('Learning Rate', color=color)
# sns.lineplot(x='epoch', y='learning_rate', data=data, ax=ax3, color=color, label='Learning Rate')
# ax3.tick_params(axis='y', labelcolor=color)

# Add combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 #+ lines3
            , labels1 + labels2 #+ labels3
            , loc='upper right')

plt.title('Training Metrics Over Epochs')
plt.tight_layout()
plt.show()