# Re-importing required library after execution state reset
import matplotlib.pyplot as plt

# Data
purge_sizes = list(range(48))
top1_accuracies = [
    17.384, 17.556, 18.933, 17.212, 18.761, 19.277, 17.384, 18.589, 20.31, 20.482, 
    20.482, 19.449, 19.966, 19.621, 18.417, 19.277, 22.892, 19.105, 19.966, 21.859, 
    20.482, 20.482, 22.892, 20.482, 21.343, 21.687, 22.547, 21.687, 19.449, 22.375, 
    22.547, 22.031, 23.064, 22.203, 22.375, 24.441, 23.236, 23.58, 22.375, 23.752, 
    23.752, 23.58, 24.441, 23.064, 24.269, 23.924, 25.129, 25.645
]

# Additional entropy data
entropy_values = [
    0.775, 0.737, 0.759, 0.745, 0.763, 0.766, 0.745, 0.763, 0.713, 0.714, 
    0.74, 0.725, 0.733, 0.729, 0.729, 0.761, 0.744, 0.735, 0.73, 0.72, 
    0.697, 0.749, 0.709, 0.729, 0.732, 0.734, 0.688, 0.747, 0.717, 0.739, 
    0.731, 0.742, 0.699, 0.716, 0.72, 0.716, 0.724, 0.7, 0.69, 0.705, 
    0.7, 0.658, 0.673, 0.677, 0.653, 0.694, 0.627, 0.674
]

# Font size settings
axis_label_fontsize = 18
axis_number_fontsize = 14
legend_fontsize = 18

# Create figure and primary axis with tight layout to remove margins
fig, ax1 = plt.subplots(figsize=(10, 7), tight_layout=True)

# Plot accuracy on primary axis
ax1.plot(purge_sizes, top1_accuracies, marker='o', linestyle='-', color='black', label='Top1 ACC')

# Highlight area from 0 to 32
ax1.axvspan(0, 32, color='yellow', alpha=0.3)

# Find max accuracy value in the range 0 to 32
max_value = max(top1_accuracies[:33])
max_index = top1_accuracies.index(max_value)

# Draw dashed lines only from the intersection to the axes
plt.axhline(y=max_value, color='green', linestyle='--', label='Max ACC [0-32]')
plt.axvline(x=max_index, color='green', linestyle='--')

# Draw reported result line at 22.89
ax1.axhline(y=22.89, color='orange', linestyle='--', label='Our Result (22.89)')

# Set x-axis ticks
ax1.set_xticks([0, 2, 4, 8, 16, 32, 48])
ax1.tick_params(axis='x', labelsize=axis_number_fontsize)
ax1.tick_params(axis='y', labelsize=axis_number_fontsize)

# Labels for accuracy axis
ax1.set_xlabel('Purge Size', fontsize=axis_label_fontsize)
ax1.set_ylabel('Top1 Acc', fontsize=axis_label_fontsize)
ax1.legend(fontsize=legend_fontsize, loc='lower center', bbox_to_anchor=(0.6, 0))
ax1.grid(True)

# Create secondary axis for entropy
ax2 = ax1.twinx()
ax2.plot(purge_sizes, entropy_values, linestyle='-', color='gray', label='Entropy')

# Labels for entropy axis
ax2.set_ylabel('Entropy', fontsize=axis_label_fontsize, color='gray')
ax2.tick_params(axis='y', labelsize=axis_number_fontsize, colors='gray')


# Save and display plot
plt.savefig('plot_purge_size_entropy.pdf')
plt.show()
