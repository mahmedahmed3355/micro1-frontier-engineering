# Case 004 Verification Tests

The Case 004 verifier evaluates shared-memory synchronization.

Static verification covers:

1. Public interface compatibility.
2. Shared-memory usage.
3. Reduction-stage structure.
4. Synchronization requirements.
5. Separation between defective and reference implementations.

Runtime verification will cover:

1. Deterministic numerical correctness.
2. Multiple reduction stages.
3. Partial blocks.
4. Multiple blocks.
5. Repeated execution.
6. Complete output validation.

The candidate output must be compared against an independent expected
result.

The verifier must not rely on one particular GPU scheduling pattern.
