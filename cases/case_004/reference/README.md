# Case 004 Reference Artifact

This directory contains the correct CUDA shared-memory reduction.

The reference establishes synchronization between reduction stages so
that shared-memory reads observe the writes on which they depend.

The reference is an independent oracle and must not be modified during
candidate evaluation.
