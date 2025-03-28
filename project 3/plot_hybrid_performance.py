import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the average timing results
df = pd.read_csv("hybrid_average_timing_results.txt")
df = df.sort_values(by='total_workers')  # Sort by total workers for clarity

# Extract data
processes = df['processes']
threads = df['threads']
total_workers = df['total_workers']
times = df['execution_time']

# Create combined labels for x-axis
labels = [f"{p}p×{t}t" for p, t in zip(processes, threads)]

# Group same total_workers with different colors
unique_workers = sorted(df['total_workers'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_workers)))

fig, ax = plt.subplots(figsize=(12, 7))

# Dictionary to hold bar positions for each worker count
positions = {}
current_pos = 0

# Create a separate subplot for each unique worker count
for i, worker_count in enumerate(unique_workers):
    subset = df[df['total_workers'] == worker_count]
    subset_labels = [f"{p}p×{t}t" for p, t in zip(subset['processes'], subset['threads'])]
    
    # Record positions for this worker count
    positions[worker_count] = list(range(current_pos, current_pos + len(subset)))
    current_pos += len(subset) + 1  # +1 for spacing between groups
    
    bars = ax.bar(positions[worker_count], subset['execution_time'], 
                 color=colors[i], label=f"{worker_count} workers")
    
    # Add time values on top of each bar
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.4f}s", ha='center', va='bottom', fontsize=9)

# Set custom tick positions and labels
all_positions = [pos for positions_list in positions.values() for pos in positions_list]
all_labels = [f"{p}p×{t}t" for p, t in zip(df['processes'], df['threads'])]

ax.set_xticks(all_positions)
ax.set_xticklabels(all_labels)

ax.set_xlabel("Configuration (processes × threads)", fontsize=12)
ax.set_ylabel("Average Execution Time (seconds)", fontsize=12)
ax.set_title("Hybrid MPI+Pthread Image Processing Performance", fontsize=14)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.legend(title="Parallel Workers")

plt.tight_layout()

# Save the plot
plt.savefig("hybrid_performance_plot.png", dpi=300, bbox_inches='tight')
plt.close()

print("Performance plot saved as 'hybrid_performance_plot.png'")
