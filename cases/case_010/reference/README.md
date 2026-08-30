# Reference Implementation

The reference implementation derives a worker-local CUDA device from rank and propagates that configuration through initialization and execution.

The tensor is made contiguous before the distributed collective and the relevant CUDA device is synchronized after the collective before the result is returned.

This establishes a deterministic configuration and execution boundary for distributed GPU workers.
