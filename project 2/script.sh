#!/bin/bash

# Compile the MPI image processing program
/opt/homebrew/bin/mpicc -o mpi_image_processor cnn_mpi.c -lm
# Clear previous results
> mpi_timing_results.txt

# Input image file
INPUT_FILE="sample3.bmp"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    echo "Please provide a valid BMP image file"
    exit 1
fi

# Initialize the results file for averages
echo "processes,execution_time" > mpi_average_timing_results.txt

# Run experiments with different process counts
for processes in 1 2 4 8 12
do
    echo -e "\nRunning with $processes processes..."
    times=()
    
    for i in {1..5}  # Run each test 5 times to get average
    do
        echo "  Run $i of 5..."
        
        # Use the correct oversubscribe option for OpenMPI
        if [ $processes -gt 8 ]; then
            # For process counts > 8, use the oversubscribe option
            mpirun --oversubscribe -np $processes ./mpi_image_processor "$INPUT_FILE"
        else
            # For process counts <= 8, use the standard command
            mpirun -np $processes ./mpi_image_processor "$INPUT_FILE"
        fi
        
        # Extract the execution time for this process count from mpi_timing_results.txt
        time_value=$(grep "^$processes " mpi_timing_results.txt | tail -1 | awk '{print $2}')
        
        if [ -n "$time_value" ]; then
            times+=($time_value)
            echo "    Time: $time_value seconds"
        else
            echo "    Error: No timing result found for $processes processes in run $i"
        fi
    done
    
    # Calculate average time
    total=0
    for t in "${times[@]}"; do
        total=$(echo "$total + $t" | bc -l)
    done
    
    if [ ${#times[@]} -gt 0 ]; then
        average=$(echo "scale=6; $total / ${#times[@]}" | bc -l)
        echo "  Average execution time for $processes processes: $average seconds"
        echo "$processes,$average" >> mpi_average_timing_results.txt
    else
        echo "  No valid timing results for $processes processes"
    fi
done

echo -e "\nAll experiments completed. Average results saved in mpi_average_timing_results.txt"

# Now create the plot using the average timing results
echo "Creating performance plot..."

cat > plot_performance.py << EOF
import matplotlib.pyplot as plt
import numpy as np

# Read the average timing results
processes = []
times = []

with open("mpi_average_timing_results.txt", "r") as file:
    next(file)  # Skip header
    for line in file:
        p, time = line.strip().split(',')
        processes.append(int(p))
        times.append(float(time))

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(processes, times, marker='o', linestyle='-', color='b', markersize=8)
plt.xlabel("Number of Processes", fontsize=12)
plt.ylabel("Average Execution Time (seconds)", fontsize=12)
plt.title("MPI Image Processing Performance", fontsize=14)
plt.grid(True)

# Annotate each data point with its value
for i, (x, y) in enumerate(zip(processes, times)):
    plt.annotate(f"{y:.4f}s", (x, y), textcoords="offset points", 
                 xytext=(0,10), ha='center')

# Save the plot
plt.savefig("mpi_performance_plot.png", dpi=300, bbox_inches='tight')
plt.close()

print("Performance plot saved as 'mpi_performance_plot.png'")
EOF

python plot_performance.py

echo "Done!"