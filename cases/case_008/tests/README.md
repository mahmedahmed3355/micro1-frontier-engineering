# Case Tests

These tests validate the distributed gradient communication contract for Case 008.

The contract checks that gradients are communicated with all_reduce and that the aggregated value is normalized before the parameter update.

The tests are deterministic and do not require a live distributed cluster or Gemini API call.
