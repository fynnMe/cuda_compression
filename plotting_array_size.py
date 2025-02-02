import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV file
data = pd.read_csv('data_uncompressed_kernel.csv', sep=';')

# Keep only the first occurrence of each array_size
data = data.drop_duplicates(subset=['array_size'], keep='first')

# Assuming your CSV has grid_size and block_size columns
# Filter for specific grid and block sizes
grid_size = 2048  # Replace with desired grid size
block_size = 32  # Replace with desired block size

filtered_data = data[
    (data['grid_size'] == grid_size) & 
    (data['block_size'] == block_size)
]

# Calculate array sizes in bytes
array_sizes_bytes = filtered_data['array_size'] * 8

# Calculate throughput (GB/s)
gigabytes_processed = array_sizes_bytes / (1024 * 1024 * 1024)
throughput = gigabytes_processed / (filtered_data['runtime'] / 1000)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(array_sizes_bytes, throughput, 'bo-', 
         linewidth=2, markersize=8)

# Add labels and title
plt.xlabel('Array Size (bytes)')
plt.ylabel('Throughput (GB/s)')
plt.title(f'Memory Throughput vs Array Size\nGrid Size: {grid_size}, Block Size: {block_size}')

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.7)

# Enable logarithmic scale for x-axis
plt.xscale('log')

# Save the plot
plt.savefig('uncompressed_array_sizes.png', dpi=500, bbox_inches='tight')

plt.show()
