# Project

SDK for running RL rollout data collection against the Token Factory API.
Users implement a rollout function and a dataset; the SDK handles concurrency, batching, model-version staleness, and API communication.

# Architecture

The runner uses a thread-based actor model: two background threads (`JobStatusTracker` and `RolloutDispatcher`) communicate with the main `RolloutRunner` event loop via a shared queue. The scheduler is a pure-logic component (no I/O) that the runner consults synchronously.

The three scheduler variants (`AnyStaleness`, `DropStale`, `Dropless`) have very different complexity. `DroplessRolloutScheduler` is the hardest to reason about — it buffers samples in a priority queue and can drain multiple batches worth of samples in a single call.

# Known invariants

- `batch_size` must be divisible by `num_samples_per_task`. Batch transitions rely on a remainder hitting exactly zero; a non-divisible config will loop forever. This is not validated in code.
- `RolloutConfig` reads env vars with prefix `TOKENFACTORY_ROLLOUT_`.

# Testing

Tests are in `tests/` using `pytest`. Runner tests mock the API client and control events via the internal queue rather than using real threads.
