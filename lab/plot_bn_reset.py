import matplotlib.pyplot as plt
import numpy as np


datasets = ["W BN Reset", "WO BN Reset"]
values = {
    "With BN Reset": [45.76, 49.13],
    "Without BN Reset": [33.00, 39.86]
}

# Define more distinct scientific colors
colors = ["#cd6155", "#5499c7"]  # More contrasting cool colors

# Define font sizes for customization
axis_label_fontsize = 18
axis_number_fontsize = 14
bar_number_fontsize = 14
legend_fontsize = 16

# Create figure
fig, ax = plt.subplots(figsize=(6, 3.5))

# X positions for bars
x = np.arange(len(datasets))
bar_width = 0.4

# Set y-axis limits
y_min = min(min(values["With BN Reset"]), min(values["Without BN Reset"])) - 5
y_max = max(max(values["With BN Reset"]), max(values["Without BN Reset"])) + 5
ax.set_ylim(y_min, y_max)

# Plot bars for both methods under each mode with distinct colors
bars1 = ax.bar(x - bar_width/2, [values["With BN Reset"][0], values["Without BN Reset"][0]], 
               width=bar_width, color=colors[0], label="BFTT3D")
bars2 = ax.bar(x + bar_width/2, [values["With BN Reset"][1], values["Without BN Reset"][1]], 
               width=bar_width, color=colors[1], label="PG-SP")

# Add text labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.2f}",
                 ha='center', va='bottom', fontsize=bar_number_fontsize, fontweight='bold')

# Draw a vertical dashed line to separate the two modes
ax.axvline(x=0.5, ymin=0, ymax=1, color='black', linestyle='dashed', linewidth=1)

# Labels and Legend
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=axis_number_fontsize)
ax.set_ylabel("Accuracy (%)", fontsize=axis_label_fontsize)
ax.tick_params(axis='y', labelsize=axis_number_fontsize)

# Move legend to the right with distinct colors
ax.legend(title="Methods", loc="upper right", fontsize=legend_fontsize, title_fontsize=legend_fontsize)

# Show plot

plt.tight_layout()
plt.savefig("bn_reset.pdf")
plt.show()
