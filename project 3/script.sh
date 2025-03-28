#!/bin/bash

# Compile the hybrid MPI+Pthread image processing program
mpicc -o hybrid_image_processor hybrid_image_processor.c -pthread

# Clear previous results
> hybrid_timing_results.txt

# Input image file
INPUT_FILE="sample3.bmp" 

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    echo "Please provide a valid BMP image file"
    exit 1
fi

# Initialize the results file for averages
echo "processes,threads,total_workers,execution_time" > hybrid_average_timing_results.txt

# Define combinations to test
# Format: "processes threads" (space-separated)
COMBINATIONS=(
    "1 1"    # 1 process x 1 thread = 1 worker (baseline)
    "1 4"    # 1 process x 4 threads = 4 workers
    "2 2"    # 2 processes x 2 threads = 4 workers
    "4 1"    # 4 processes x 1 thread = 4 workers
    "2 4"    # 2 processes x 4 threads = 8 workers
    "4 2"    # 4 processes x 2 threads = 8 workers
    "8 1"    # 8 processes x 1 thread = 8 workers
    "3 4"    # 3 processes x 4 threads = 12 workers
    "4 3"    # 4 processes x 3 threads = 12 workers
    "6 2"    # 6 processes x 2 threads = 12 workers
)

# Run experiments for each combination
for COMBO in "${COMBINATIONS[@]}"; do
    # Extract processes and threads
    PROCESSES=$(echo $COMBO | cut -d' ' -f1)
    THREADS=$(echo $COMBO | cut -d' ' -f2)
    TOTAL_WORKERS=$((PROCESSES * THREADS))
    
    echo -e "\n=========================================================="
    echo "Running with $PROCESSES processes × $THREADS threads ($TOTAL_WORKERS total workers)"
    echo "=========================================================="
    times=()
    
    for i in {1..5}  # Run each test 5 times to get average
    do
        echo "  Run $i of 5..."
        
        # Use oversubscribe option if processes > 8
        if [ $PROCESSES -gt 8 ]; then
            mpirun --oversubscribe -np $PROCESSES ./hybrid_image_processor "$INPUT_FILE" $THREADS
        else
            mpirun -np $PROCESSES ./hybrid_image_processor "$INPUT_FILE" $THREADS
        fi
        
        # Extract the execution time from hybrid_timing_results.txt
        # Format in file: processes threads execution_time
        time_value=$(grep "^$PROCESSES $THREADS " hybrid_timing_results.txt | tail -1 | awk '{print $3}')
        
        if [ -n "$time_value" ]; then
            times+=($time_value)
            echo "    Time: $time_value seconds"
        else
            echo "    Error: No timing result found for $PROCESSES processes and $THREADS threads in run $i"
        fi
    done
    
    # Calculate average time
    total=0
    for t in "${times[@]}"; do
        total=$(echo "$total + $t" | bc -l)
    done
    
    if [ ${#times[@]} -gt 0 ]; then
        average=$(echo "scale=6; $total / ${#times[@]}" | bc -l)
        echo "  Average execution time: $average seconds"
        echo "$PROCESSES,$THREADS,$TOTAL_WORKERS,$average" >> hybrid_average_timing_results.txt
    else
        echo "  No valid timing results for this combination"
    fi
done

echo -e "\nAll experiments completed. Average results saved in hybrid_average_timing_results.txt"

# create the plot using the average timing results
echo "Creating performance plot..."

cat > plot_hybrid_performance.py << EOF
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
EOF

python plot_hybrid_performance.py

echo "Done!"