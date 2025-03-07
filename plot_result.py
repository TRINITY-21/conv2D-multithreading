import matplotlib.pyplot as plt # type: ignore

# Read the timing results from the file
threads = []
times = []

with open("timing_results2.txt", "r") as file:
    for line in file:
        t, time = line.split()
        threads.append(int(t))
        times.append(float(time))

# Plot the results
plt.figure(figsize=(8, 6))  # Set figure size
plt.plot(threads, times, marker='o', linestyle='-', color='b', markersize=8)

# Labels and title
plt.xlabel("Number of Threads", fontsize=12)
plt.ylabel("Execution Time (seconds)", fontsize=12)
plt.title("Image Processing Performance", fontsize=14)
plt.grid(True)

# Save the plot as an image
plt.savefig("performance_plot.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print("Performance plot saved as 'performance_plot.png'")
