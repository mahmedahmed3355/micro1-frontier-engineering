# Defective Implementation

The implementation intentionally leaks `device_output` on the successful execution path.

The same allocation is correctly released on several failure paths.

This is intentional benchmark behavior.

The target is control-flow-aware resource-lifetime reasoning.
