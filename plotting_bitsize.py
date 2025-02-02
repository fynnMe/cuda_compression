import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV file
data = pd.read_csv('compressed_configs.csv', sep=';')

# Keep only the first occurrence of each array_size and order by bitsize
data = data.drop_duplicates(subset=['bit_size'], keep='first').sort_values('bit_size')

# Filter for specific grid and block sizes
grid_size = 2048
block_size = 1024

filtered_data = data[
    (data['grid_size'] == grid_size) & 
    (data['block_size'] == block_size)
]

# Calculate actual speedup
speedup = (8.367317 / filtered_data['runtime']) * (64 / filtered_data['bit_size'])

# Calculate optimal speedup
speedup_optimal = 64 / filtered_data['bit_size']

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['bit_size'], speedup, 'b-',
         linewidth=2, markersize=8, label='Actual Speedup')
plt.plot(filtered_data['bit_size'], speedup_optimal, 'r:',
         linewidth=2, markersize=8, label='Optimal Speedup')

# Add labels and title
plt.xlabel('Bit Size')
plt.ylabel('Speedup')
plt.title(f'Naive Compressed Kernel Speedup vs Static Bit Size of Elements in uint64 \nGrid Size: {grid_size}, Block Size: {block_size}')

# Uncomment to zoom in
plt.ylim([0, 13])

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Save the plot
plt.savefig('speedup_vs_bitsize.png', dpi=500, bbox_inches='tight')

plt.show()
