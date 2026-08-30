# Case 002 Source Artifact

This directory contains the defective implementation used by the
benchmark.

The implementation intentionally contains a host/device transfer
completion and lifetime defect.

The public run_transfer interface is part of the case contract and must
remain compatible with the reference implementation.

This source is evaluated against deterministic transfer workloads.
