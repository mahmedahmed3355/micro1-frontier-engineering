# Defective Implementation

This directory contains the intentionally defective compound distributed GPU implementation.

The configuration incorrectly collapses every worker onto the generic CUDA device instead of deriving the worker-local CUDA device from rank.

The execution path also omits explicit GPU completion synchronization after the distributed collective.

The benchmark is intended to determine whether an agent can trace configuration from worker construction through distributed initialization and GPU execution.
