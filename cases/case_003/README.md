# Case 003 — CUDA Stream and Event Dependency

## Task

Repair a CUDA pipeline that performs a host-to-device transfer in one
CUDA stream and consumes the transferred data in another CUDA stream.

The public function must return only after the requested output has been
fully produced.

## Failure Mode

The defective implementation places the producer operations in one
stream and the consumer operations in another stream without establishing
an explicit dependency between them.

CUDA stream ordering is local to an individual stream.

The consumer stream therefore cannot assume that work submitted to the
producer stream has completed.

## Expected Behavior

The implementation must establish an explicit producer-to-consumer
dependency.

The required ordering is:

1. Host-to-device transfer completes.
2. Consumer stream is allowed to consume the device buffer.
3. Kernel execution completes.
4. Device-to-host transfer completes.
5. The public function returns.

The dependency must be expressed using CUDA stream/event primitives or
another CUDA mechanism that provides equivalent ordering guarantees.

## Engineering Constraints

Preserve the public run_stream_dependency interface.

Keep the producer and consumer operations in their intended separate
streams.

Do not replace the asynchronous pipeline with a CPU implementation.

Do not rely on implicit global synchronization.

Do not assume that two different streams execute in submission order.

Do not use host-side sleeps or timing assumptions as synchronization.

## Correctness Requirements

For every valid input size, the output must equal the input plus the
provided scalar value.

The implementation must remain correct when operations overlap.

Repeated execution must remain deterministic.

The device buffer must remain alive until all operations using it have
completed.

## Evaluation

The verifier will inspect both functional behavior and the stream
dependency structure.

The test design must include repeated execution and workloads large
enough to make asynchronous overlap meaningful.

The candidate must not pass solely because a particular GPU happens to
execute the producer before the consumer.

## Engineering Goal

The objective is to correctly model an inter-stream dependency and make
the producer-to-consumer relationship explicit, deterministic, and safe.
