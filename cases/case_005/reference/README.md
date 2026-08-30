# Reference Implementation

The reference releases both device allocations on the successful path.

`device_output` is released before `device_input`, and both releases occur after the final device-to-host transfer.

Cleanup failures remain observable through the returned status.
