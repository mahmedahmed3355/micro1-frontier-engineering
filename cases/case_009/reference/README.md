# Reference Implementation

This directory contains the correct asynchronous GPU worker.

The reference awaits the asynchronous work and explicitly synchronizes the relevant CUDA device before transferring a CUDA result to host memory.

The synchronization establishes the required GPU completion boundary.
