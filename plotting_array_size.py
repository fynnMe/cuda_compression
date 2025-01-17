import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV file
data = pd.read_csv('compressed_array_size.csv')
array_sizes = data['array_size']

# Calculate array sizes in bytes
array_sizes_bytes = array_sizes * 8  # Convert to bytes

# Calculate throughput (MB/s)
gigabytes_processed = array_sizes_bytes / (1024 * 1024 * 1024)  # Convert to GB
throughput = gigabytes_processed / (data['runtime'] / 1000)  # GB/s

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(array_sizes_bytes, throughput, 'bo-', linewidth=2, markersize=8)

# Add labels and title
plt.xlabel('Array Size (bytes)')
plt.ylabel('Throughput (GB/s)')
plt.title('Memory Throughput vs Array Size')

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Enable logarithmic scale for x-axis
plt.xscale('log')

# Save the plot
plt.savefig('compressed_array_sizes.png', dpi=500, bbox_inches='tight')

plt.show()
