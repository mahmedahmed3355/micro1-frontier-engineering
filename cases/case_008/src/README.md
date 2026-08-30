# Defective Implementation

This directory contains the intentionally defective implementation for Case 008.

The defect is an incomplete distributed gradient communication step. Worker-local gradients are communicated, but the resulting aggregate is applied without normalization by the distributed world size.

The defect must remain intentional so that the benchmark evaluates whether an agent can identify the communication contract violation.
