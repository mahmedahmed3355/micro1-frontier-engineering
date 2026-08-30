# Case 003 Reference Artifact

This directory contains the correct CUDA stream pipeline.

The reference explicitly establishes the dependency between the producer
stream and consumer stream before the consumer accesses the produced
device data.

The reference is an independent oracle and must not be modified during
candidate evaluation.
