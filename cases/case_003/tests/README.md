# Case 003 Verification Tests

The Case 003 verifier checks stream dependency correctness.

The verification plan covers:

1. Public interface compatibility.
2. Producer and consumer stream separation.
3. Explicit inter-stream dependency.
4. Correct kernel ordering.
5. Correct device-to-host completion.
6. Device-buffer lifetime.
7. Deterministic numerical output.
8. Repeated execution.

Static checks must confirm that the defective artifact lacks the required
producer-to-consumer dependency and that the reference contains one.

Runtime checks must compare candidate output with an independent expected
result.

The candidate must not depend on accidental scheduling behavior.
