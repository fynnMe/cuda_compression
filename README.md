# Highly Parallel Programming Of GPUs - Course Project Compression

## Group Members
- Theo Reichert
- Fynn Meffert

## Research
- [RTX 8000 Nvidia](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-8000-us-nvidia-946977-r1-web.pdf)
    - Turing Architecture
    - 4,608 CUDA cores
- [RTX 8000 TechPowerUp](https://www.techpowerup.com/gpu-specs/quadro-rtx-8000.c3306)
    - 72 SMs running 64 threads each
- [Turing Architecture Information](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)
    - max. 16 thread blocks per SM (-> $72\cdot16=1152$ maximum number of blocks)

## Usage

```bash
# compile .cu files
nvcc uncompressed.cu -o uncompressed
nvcc compressed.cu -o compressed

# TODO not run binaries via python script
#   `evaluation.py`
# run the binaries directly
#   they create .csv files
uncompressed NUM_ELEMENTS BLOCK_SIZE GRID_SIZE
compressed NUM_ELEMENTS BLOCK_SIZE GRID_SIZE
```

```bash
# set up python virtual environment
#  optional: https://docs.python.org/3/library/venv.html
python3 -m venv virtualenvironment_python
source ./virtualenvironment_python/bin/activate
pip install pandas seaborn matplotlib

# generate heatmap from .csv files 
#   via python script `plotting_configs.py`
python3 plotting_configs.py
```
