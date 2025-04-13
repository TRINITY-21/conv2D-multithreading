#!/bin/bash

# Check if libomp is installed
if ! brew list --formula | grep -q "^libomp$"; then
    echo "libomp not found. Installing via Homebrew..."
    brew install libomp
fi

# Get the path to libomp
LIBOMP_PATH=$(brew --prefix libomp)

# Create a symbolic link in a standard include directory (only once)
# This lets you use #include <omp.h> without modifications
if [ ! -f "/usr/local/include/omp.h" ]; then
    echo "Creating symbolic link for omp.h in standard include path..."
    sudo mkdir -p /usr/local/include
    sudo ln -s $LIBOMP_PATH/include/omp.h /usr/local/include/omp.h
fi

# Compile the OpenMP image processing program
echo "Compiling with OpenMP support..."
clang -Xpreprocessor -fopenmp -L$LIBOMP_PATH/lib -lomp -o openmp_image_processor cnn_openmp.c -lm

# Rest of your script remains the same...

# Clear previous results
> openmp_timing_results.txt

# Input image file
INPUT_FILE="sample3.bmp"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    echo "Please provide a valid BMP image file"
    exit 1
fi

# Initialize the results file for averages
echo "threads,execution_time" > openmp_average_timing_results.txt

# Run experiments with different thread counts
for threads in 1 2 4 8 12
do
    echo -e "\nRunning with $threads threads..."
    times=()
    
    for i in {1..5}  # Run each test 5 times to get average
    do
        echo "  Run $i of 5..."
        
        # Run the OpenMP program with specified thread count
        ./openmp_image_processor "$INPUT_FILE" $threads
        
        # Extract the execution time from results file
        time_value=$(grep "^$threads " openmp_timing_results.txt | tail -1 | awk '{print $2}')
        
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
        echo "$threads,$average" >> openmp_average_timing_results.txt
    else
        echo "  No valid timing results for $threads threads"
    fi
done

echo -e "\nAll experiments completed. Average results saved in openmp_average_timing_results.txt"

# Now create the plot using the average timing results
echo "Creating performance plot..."

cat > plot_openmp_performance.py << EOF
import matplotlib.pyplot as plt
import numpy as np

# Read the average timing results
threads = []
times = []

with open("openmp_average_timing_results.txt", "r") as file:
    next(file)  # Skip header
    for line in file:
        t, time = line.strip().split(',')
        threads.append(int(t))
        times.append(float(time))

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(threads, times, marker='o', linestyle='-', color='b', markersize=8)
plt.xlabel("Number of Threads", fontsize=12)
plt.ylabel("Average Execution Time (seconds)", fontsize=12)
plt.title("OpenMP Image Processing Performance", fontsize=14)
plt.grid(True)

# Annotate each data point with its value
for i, (x, y) in enumerate(zip(threads, times)):
    plt.annotate(f"{y:.4f}s", (x, y), textcoords="offset points", 
                 xytext=(0,10), ha='center')

# Save the plot
plt.savefig("openmp_performance_plot.png", dpi=300, bbox_inches='tight')
plt.close()

print("Performance plot saved as 'openmp_performance_plot.png'")
EOF

python3 plot_openmp_performance.py

echo "Done!"