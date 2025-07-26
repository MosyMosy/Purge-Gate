import matplotlib.pyplot as plt
import numpy as np

# Data for execution time (Boxplot)
time_data = [[8.122, 31.84, 10.632, 5.703], [221.008, 239.454, 230.821, 4.484]]
methods = ["PG-SP", "BFTT3D"]

# Data for memory complexity (Bar plot)
memory_values = [1331, 7443]

# Define **consistent** colors for methods
colors = ["#5499c7","#cd6155"]

# Font size settings for customization
title_fontsize = 18       # Font size for plot titles
axis_label_fontsize = 18  # Font size for axis labels (Y-axis)
tick_fontsize = 14        # Font size for axis numbers (ticks)
value_fontsize = 16       # Font size for displayed mean values

# Create figure with two subplots (side-by-side)
fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(7, 4), width_ratios=[2.5, 1])

# ---- Execution Time Boxplot (Left) ----
positions = [0.5, 2]  # More spacing between methods
box_width = 0.8  # Wider boxplots

# Create boxplots without the median line
box1 = ax_time.boxplot(time_data[0], vert=True, patch_artist=True, positions=[positions[0]], 
                        labels=["PG-SP"], widths=box_width, medianprops={'visible': False}, 
                        boxprops=dict(color='black'), flierprops={'marker': None})
box2 = ax_time.boxplot(time_data[1], vert=True, patch_artist=True, positions=[positions[1]], 
                        labels=["BFTT3D"], widths=box_width, medianprops={'visible': False}, 
                        boxprops=dict(color='black'), flierprops={'marker': None})

# Apply **consistent colors** to boxplots
for patch in box1['boxes']:
    patch.set_facecolor(colors[0])  # PG-SP color
for patch in box2['boxes']:
    patch.set_facecolor(colors[1])  # BFTT3D color

# Calculate and plot mean values as **dashed black lines**
means = [np.mean(time_data[0]), np.mean(time_data[1])]
ax_time.plot([positions[0] - 0.45, positions[0] + 0.45], [means[0], means[0]], 'k--', linewidth=1.5)
ax_time.plot([positions[1] - 0.45, positions[1] + 0.45], [means[1], means[1]], 'k--', linewidth=1.5)

# Annotate mean values beside the dashed lines
ax_time.text(positions[0] + 0.5, means[0], f"{means[0]:.2f}", verticalalignment='center', 
             fontsize=value_fontsize, color='black')
ax_time.text(positions[1] - 1.2, means[1], f"{means[1]:.2f}", verticalalignment='center', 
             fontsize=value_fontsize, color='black')

# Adjust spacing and width
ax_time.set_xticks(positions)
ax_time.set_xticklabels(methods, fontsize=tick_fontsize)
ax_time.set_xlim(0, 3)  # Ensuring spacing remains
ax_time.set_ylim(0, 250)
ax_time.set_ylabel("Execution Time (ms)", fontsize=axis_label_fontsize)
ax_time.set_title("Batch Time", fontsize=title_fontsize)

# ---- Memory Complexity Barplot (Right) ----
ax_mem.bar([0.9, 1.1], memory_values, color=colors, width=0.15)  # Use same colors

# Set labels for memory plot
ax_mem.set_xticks([0.9, 1.1])
ax_mem.set_xticklabels(methods, fontsize=tick_fontsize)
ax_mem.set_ylabel("Memory (MiB)", fontsize=axis_label_fontsize)
ax_mem.set_title("GPU Memory", fontsize=title_fontsize)

# Adjust tick labels size
ax_time.tick_params(axis='y', labelsize=tick_fontsize)
ax_mem.tick_params(axis='y', labelsize=tick_fontsize)

# Adjust layout for compactness
fig.tight_layout()

# Show plot
plt.savefig("time_memory_plot.pdf")
plt.show()
