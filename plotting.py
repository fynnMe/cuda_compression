# python3 plotting.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('kernel_performance.csv')

# Create a pivot table for the heatmap
pivot_table = df.pivot(index='block_size', 
                      columns='grid_size', 
                      values='runtime')

# Set up the plot style
plt.figure(figsize=(12, 8))

# Create heatmap
sns.heatmap(pivot_table, 
            annot=True,  # Show values in cells
            fmt='.4f',   # Format to 2 decimal places
            cmap='YlOrRd',  # Yellow to Orange to Red colormap
            cbar_kws={'label': 'Runtime (ms)'},
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
plt.savefig('kernel_performance_heatmap.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
