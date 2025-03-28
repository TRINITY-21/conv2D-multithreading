#!/bin/bash

# Compile the image processing program
gcc -o image_processor cnn.c -pthread -lm

# Clear previous results
echo "" > timing_results.txt

# Input image file
INPUT_FILE="lena.bmp"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    echo "Please provide a valid BMP image file"
    exit 1
fi

# Initialize the results file for averages
echo "threads,execution_time" > average_timing_results.txt

# Run experiments with 1, 3, 6, 9, 12 threads
for threads in 1 3 6 9 12
do
    echo -e "\nRunning with $threads threads..."
    times=()
    
    for i in {1..5}  # Run each test 5 times to get average
    do
        echo "  Run $i of 5..."
        # Run the image processor and capture the execution time
        ./image_processor "$INPUT_FILE"
        
        # Extract the execution time for this thread count from timing_results.txt
        time_value=$(grep "^$threads " timing_results.txt | tail -1 | awk '{print $2}')
        
        if [ -n "$time_value" ]; then
            times+=($time_value)
            echo "    Time: $time_value seconds"
        else
            echo "    Error: No timing result found for $threads threads in run $i"
        fi
    done
    
    # Calculate average time
    total=0
    for t in "${times[@]}"; do
        total=$(echo "$total + $t" | bc -l)
    done
    
    if [ ${#times[@]} -gt 0 ]; then
        average=$(echo "scale=6; $total / ${#times[@]}" | bc -l)
        echo "  Average execution time for $threads threads: $average seconds"
        echo "$threads,$average" >> average_timing_results.txt
    else
        echo "  No valid timing results for $threads threads"
    fi
done

echo -e "\nAll experiments completed. Average results saved in average_timing_results.txt"

# create the plot using the average timing results
echo "Creating performance plot..."

cat > plot_performance.py << EOF
import matplotlib.pyplot as plt
import numpy as np

# Read the average timing results
threads = []
times = []

with open("average_timing_results.txt", "r") as file:
    next(file)  # Skip header
    for line in file:
        t, time = line.strip().split(',')
        threads.append(int(t))
        times.append(float(time))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(threads, times, marker='o', linestyle='-', color='b', markersize=8)

# Add labels and title
plt.xlabel("Number of Threads", fontsize=12)
plt.ylabel("Average Execution Time (seconds)", fontsize=12)
plt.title("Image Processing Performance (Average of 5 runs)", fontsize=14)
plt.grid(True)

# Annotate each data point with its value
for i, (x, y) in enumerate(zip(threads, times)):
    plt.annotate(f"{y:.4f}s", (x, y), textcoords="offset points", 
                 xytext=(0,10), ha='center')

# Save the plot
plt.savefig("performance_plot.png", dpi=300, bbox_inches='tight')
plt.show()

print("Performance plot saved as 'performance_plot.png'")
EOF

python plot_performance.py

echo "Done!"