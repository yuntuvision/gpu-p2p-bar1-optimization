# GPU BAR1 / P2P Optimization Notes

This repository contains the measured, workload-level optimization plan for two
8× RTX 4090 nodes using a custom NVIDIA 610.43.02 driver build.

It is intentionally a separate runbook repository. It does not contain driver
binaries, firmware images, credentials, host addresses, or automatic system
mutation scripts.

## Current result

- Every GPU reports a 64 GiB BAR1 aperture.
- CUDA P2P content copies pass in all 56 off-diagonal directions on both nodes.
- Same-group copy-engine bandwidth is about 26.3 GB/s per direction.
- Cross-group single-pair bandwidth is about 22–23 GB/s when only the active
  reciprocal peer mapping is enabled.
- Four fixed cross-group bridge pairs provide about 56–62 GB/s aggregate one-way
  throughput when transfers are striped and scheduled concurrently.

## Recommended operating model

Use topology-aware, pair-scoped P2P mappings instead of enabling all 56 peer
relationships for copy-engine traffic. Start with these bridge pairs:

```text
0 <-> 4
1 <-> 5
2 <-> 6
3 <-> 7
```

Stripe large transfers over the four pairs. This avoids the severe reverse-
direction degradation observed when every peer mapping is active.

## Important limits

- 26 GB/s is the same-group PCIe result; it is not a guaranteed cross-CPU-socket
  result.
- One node has a BIOS/NUMA topology problem that makes the all-peer state
  asymmetric.
- On the other node, GPU1 is trained at PCIe 4.0 x8. It needs physical/firmware
  remediation before it can match x16 paths.
- The serial two-hop relay experiment was slower than direct copy. A relay is not
  recommended without an application-integrated, event-synchronized pipeline.

## Contents

- [`docs/solution-plan.md`](docs/solution-plan.md): options, trade-offs, and rollout plan.
- [`docs/validation-results.md`](docs/validation-results.md): sanitized measured results.
- [`docs/reproduction.md`](docs/reproduction.md): safe benchmark procedure.
- [`scripts/cuda-p2p-pair-benchmark.py`](scripts/cuda-p2p-pair-benchmark.py): one-way and bidirectional pair test.
- [`scripts/cuda-p2p-relay-benchmark.py`](scripts/cuda-p2p-relay-benchmark.py): direct versus serial relay probe.

All scripts are diagnostic. Run them only when the GPUs are idle and verify the
process list before and after each test.
