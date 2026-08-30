# Reference Implementation

This directory contains the correct reference implementation for Case 008.

The reference communicates the gradient across workers and normalizes the aggregated gradient by the distributed world size before applying the parameter update.

This implementation is the behavioral oracle for the case.
