# Notes On Code Decisions

## `uncompressed.cu`
- grid striding kernel (no monolithic kernel) since the arrays are large
- no further optimizations to the kernel (esp. memory usage) since there is no data reuse
- "base case"