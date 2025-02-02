#!/bin/bash
for i in {1..64}
do
    # First compile
    nvcc -DBITSIZE=$i compressed.cu -o compressed
    
    # If compilation successful, run the program and pipe to Python
    if [ $? -eq 0 ]; then
        python3 evaluation.py
        wait
    else
        echo "Compilation failed for BITSIZE=$i"
    fi
done
