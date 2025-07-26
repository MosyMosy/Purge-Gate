# Re-import necessary libraries since execution state was reset
import matplotlib.pyplot as plt
import numpy as np

# Data for PG-SP method
batch_sizes = np.array([2, 4, 8, 16, 32,]) # 64, 128])
pg_sp_acc = np.array([43.82, 55.92, 61.10, 63.27, 64.12]) #, 64.54, 64.76])

tent_acc =  np.array([4.58, 8.80, 19.70, 36.86, 54.73]) #, 60.44, 62.58])

# Define font size variables
axis_label_fontsize = 18
axis_number_fontsize = 14
legend_fontsize = 18

# Creating the plot again with defined font sizes
plt.figure(figsize=(6, 3))

# Plot PG-SP
plt.plot(batch_sizes, pg_sp_acc, marker='o', linestyle='-', linewidth=2, markersize=6, label='PG-SP', color="#5499c7")
# Plot TENT
plt.plot(batch_sizes, tent_acc, marker='s', linestyle='--', linewidth=2, markersize=6, label='TENT', color="#cd6155")

# Labels with controlled font size
plt.xlabel("Batch Size", fontsize=axis_label_fontsize)
plt.ylabel("Top-1 Acc (%)", fontsize=axis_label_fontsize)

# Adjusting axis number font sizes
plt.xticks(fontsize=axis_number_fontsize)
plt.yticks(fontsize=axis_number_fontsize)

# Customizing legend font size
plt.legend(fontsize=legend_fontsize, loc='lower right')

# Log scale for x-axis
plt.xscale("log", base=2)
plt.grid(True, linestyle="--", alpha=0.6)

# Show updated plot
plt.tight_layout()
plt.savefig("batch_size_plot.pdf")
plt.show()
