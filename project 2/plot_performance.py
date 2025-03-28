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
