#!/usr/bin/env python3
import subprocess
import itertools

# Define the ranges for each parameter
num_elements_range = [134217728]
block_size_range = [2**i for i in range(5, 11)]
grid_size_range = [2**i for i in range(1, 12)]

# Loop over all combinations using itertools.product
for num_elements, block_size, grid_size in itertools.product(
    num_elements_range,
    block_size_range,
    grid_size_range
):
    # Construct the command
    command = [
        "./uncompressed",
        str(num_elements),
        str(block_size),
        str(grid_size)
    ]
    
    try:
        # Run the executable and capture output
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Successfully ran with parameters:")
        print(f"  num_elements: {num_elements}")
        print(f"  block_size: {block_size}")
        print(f"  grid_size: {grid_size}")
        print(f"Output: {result.stdout}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running command with parameters:")
        print(f"  num_elements: {num_elements}")
        print(f"  block_size: {block_size}")
        print(f"  grid_size: {grid_size}")
        print(f"Error: {e}")
