# Case 004 Source Artifact

This directory contains the intentionally defective CUDA reduction.

The implementation uses shared memory for a tree reduction but omits the
required synchronization between dependent reduction stages.

The public run_reduction interface is part of the case contract.
