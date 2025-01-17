import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('uncompressed_configs.csv', sep=';')

# Calculate throughput (MB/s)
num_elements = 134217728 # TODO hardcoded here but not in uncompressed.cu
bytes_processed = num_elements * 8
gigabytes_processed = bytes_processed / (1024 * 1024 * 1024)  # Convert to GB
df['throughput'] = gigabytes_processed / (df['runtime'] / 1000)  # GB/s

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
            cbar_kws={'label': 'Throughput (GB/s)'}
            )

# Customize the plot
plt.title('CUDA Kernel Performance: Block Size vs Grid Size')
plt.xlabel('Grid Size')
plt.ylabel('Block Size')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('uncompressed_configs.png', dpi=500, bbox_inches='tight')

# Show the plot
plt.show()
