# Defective Implementation

This directory contains the intentionally defective implementation for Case 007.

The defect models a distributed training synchronization problem in which the worker-local loss is not explicitly synchronized before the optimization boundary.

The defect must remain intentional. The benchmark should determine whether an agent can identify the missing distributed coordination.
