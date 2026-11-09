# Solution plan

## Intent

Improve inter-group GPU communication without claiming that a BAR1/P2P change
increases the RTX 4090's CUDA-core throughput. The practical target is higher
aggregate communication throughput and removal of the 10 GB/s reverse-direction
degradation.

## Goals

1. Preserve the already verified 64 GiB BAR1 and 56/56 P2P content-copy state.
2. Keep each active cross-group copy near the measured 22–23 GB/s ceiling.
3. Scale aggregate group-to-group traffic with independent bridge pairs.
4. Avoid blind PCIe register writes, firmware flashing, or persistent changes.

## Non-goals

- Claiming 26 GB/s for a cross-CPU-socket single direction.
- Unlocking CUDA cores, tensor-core count, or stock RTX 4090 TFLOPS.
- Replacing a workload's collective-communication library without its topology
  and correctness tests.

## Approaches considered

### A. BIOS/BMC topology correction

Expose two NUMA domains, disable node interleaving, and verify NPS/xGMI width and
speed settings. This is the only route that can change the platform's underlying
topology, but it needs a maintenance reboot and working BMC/BIOS access.

### B. Pair-scoped bridge scheduling — recommended

Enable only the peer mappings required by the current transfer plan and stripe
traffic over four cross-group pairs. This is reversible, does not touch firmware,
and is the only approach measured to improve the observed reverse-direction
degradation: approximately 10 GB/s in the all-peer state became approximately
22.6 GB/s for an isolated active pair.

### C. Two-hop relay through a healthy x16 GPU

Route an x8-limited GPU through a local x16 GPU. The safe serial prototype was
only 2.35–2.38 GB/s effective, so it is rejected as a default. It could only be
reconsidered as a workload-integrated, pipelined design with explicit CUDA-event
synchronization and end-to-end correctness checks.

### D. Blind driver/PCIe register changes

Rejected. The measured failures are explained by NUMA exposure, peer-map fan-out,
and one physical x8 link. Unverified register writes add reset/data-corruption
risk without evidence of a higher ceiling.

## Rollout plan

1. Detect idle GPUs and record BAR1, driver, PCIe links, and P2P content health.
2. Select four bridge pairs from the topology matrix.
3. Enable reciprocal peer access only for the active pairs.
4. Stripe transfers and schedule opposite directions concurrently where possible.
5. Re-run content validation, bandwidth, and application-level correctness tests.
6. Revert to the normal communicator setup if any direction regresses or any GPU
   reports an Xid/AER error.

## Acceptance criteria

- 56/56 supported content directions remain correct.
- No compute jobs, Xid, fatal AER, or hardware-error messages are left behind.
- Active cross-group pairs remain within the expected 21–23 GB/s range.
- Aggregate throughput is reported as the sum of synchronized active pairs, not
  mislabeled as the bandwidth of one link.
