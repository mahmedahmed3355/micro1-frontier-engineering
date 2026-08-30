# Case Tests

These tests validate the static distributed-training contract for Case 007.

The contract checks that distributed synchronization and aggregation occur after the backward pass and before the optimizer step.

The tests are deterministic and do not require a live distributed cluster or Gemini API call.
