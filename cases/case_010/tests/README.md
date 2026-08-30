# Case Tests

These tests validate the compound distributed GPU configuration contract.

The contract covers:

- rank-derived CUDA device selection
- configuration propagation
- NCCL distributed initialization
- worker-local CUDA device assignment
- tensor placement
- contiguous collective input
- distributed all-reduce
- explicit CUDA completion synchronization
- ordering across the complete execution path

The tests are static and deterministic. They do not require a live Gemini API call or actual multi-process CUDA execution.
