import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('data_compressed_kernel.csv', sep=';')

# Filter for only 134217728 elements and keep first occurrence of duplicates
df = df[df['array_size'] == 134217728].drop_duplicates(subset=['block_size', 'grid_size'], keep='first')

# Calculate throughput (GiB/s)
bytes_per_elements = 8
num_arrays = 3
bytes_processed = 134217728 * bytes_per_elements * num_arrays
gigabytes_processed = bytes_processed / (1024 * 1024 * 1024)  # Convert to GiB
df['throughput'] = gigabytes_processed / (df['runtime'] / 1000)  # GiB/s

# Create a pivot table for the heatmap with throughput
pivot_table = df.pivot(index='block_size',
                      columns='grid_size',
                      values='throughput')

# Sort the index in descending order
pivot_table = pivot_table.sort_index(ascending=False)

# Set up the plot style
plt.figure(figsize=(12, 8))

# Create heatmap
sns.heatmap(pivot_table,
            annot=True,  # Show values in cells
            fmt='.2f',   # Float with 2 decimal places
            cmap='YlOrRd',  # Yellow to Orange to Red colormap
            cbar_kws={'label': 'Throughput (GiB/s)'}
            )

# Customize the plot
plt.title('Naive Compressed CUDA Kernel Performance: Block Size vs Grid Size\n(Arrays of 1 GiB Size Each)')
plt.xlabel('Grid Size')
plt.ylabel('Block Size')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('compressed_configs.png', dpi=500, bbox_inches='tight')

# Show the plot
plt.show()
