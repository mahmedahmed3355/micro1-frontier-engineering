# Case Tests

These tests validate the asynchronous GPU worker contract.

The contract checks:

- asynchronous task creation
- awaiting worker completion
- CUDA-result detection
- device synchronization before host transfer
- preservation of the synchronization defect in the source implementation

The tests are deterministic and do not require a live Gemini API call or actual CUDA execution.
