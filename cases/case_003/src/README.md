# Case 003 Source Artifact

This directory contains the intentionally defective CUDA stream pipeline.

The producer uses one CUDA stream and the consumer uses another stream.

The implementation intentionally omits the explicit dependency required
between those streams.

The public run_stream_dependency interface must remain unchanged.
