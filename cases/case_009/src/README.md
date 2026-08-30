# Defective Implementation

This directory contains the intentionally defective asynchronous GPU worker.

The defect occurs at the boundary between asynchronous task completion and host-side consumption of a CUDA tensor. The worker awaits the Python task but does not explicitly synchronize the CUDA device before transferring the result to the CPU.

The defect must remain intentional so the benchmark can evaluate whether an agent identifies the missing GPU completion boundary.
