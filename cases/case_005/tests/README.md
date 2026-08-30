# Case Tests

The contract tests are static source-level checks.

They intentionally distinguish successful-path cleanup from failure-path cleanup.

The defective source must contain output cleanup on failure paths while omitting it from the successful cleanup sequence.

The reference must release both allocations on the successful path in deterministic order.
