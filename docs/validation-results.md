# Sanitized validation results

Test date: 2026-11-09. Both nodes were idle before and after testing. No driver,
kernel, BIOS, BMC, or firmware setting was changed during this investigation.

## Node A (8× RTX 4090)

- Driver: 610.43.02.
- BAR1: 65536 MiB on all eight GPUs.
- P2P: 56/56 supported directions; byte-for-byte content checks passed.
- Same-group bandwidth: 26.285–26.311 GB/s.
- Cross-group all-peer state: approximately 9.8–22.7 GB/s depending on direction.
- Pair-scoped state: an isolated reverse-direction pair reached 22.65–22.67 GB/s.
- One pair bidirectional: approximately 41.65–41.67 GB/s aggregate.
- Four simultaneous one-way bridges: approximately 57.5–61.6 GB/s summed.
- CPU package affinity did not change the slow direction, ruling out ordinary
  process pinning as a fix.

The platform exposes both CPU sockets as one logical NUMA node. That is the main
explanation for the asymmetry seen when all peer mappings are enabled.

## Node B (8× RTX 4090)

- Driver: 610.43.02.
- BAR1: 65536 MiB on all eight GPUs.
- P2P: 56/56 supported directions; byte-for-byte content checks passed.
- Same-group median: approximately 26.308 GB/s.
- Typical cross-group paths: approximately 21.8–22.7 GB/s.
- GPU1 is trained at PCIe 4.0 x8; paths through it are approximately 13.15–13.19
  GB/s. Other GPU links are x16.
- Four simultaneous one-way bridges: approximately 56.8–56.9 GB/s summed.
- Serial GPU1→GPU0→GPU4 relay: approximately 2.35 GB/s effective; rejected.

The x8 link is a physical/firmware link-training issue. A standard guarded PCIe
retrain did not change it; cold power-off, reseating, riser/slot inspection, or
BIOS inspection is required before re-testing.

## NCCL note

The preinstalled NCCL package initially failed with `CUDA driver is a stub
library`. A temporary CUDA-compatible NCCL build passed 8-GPU correctness, but
the tested all-reduce path was only about 1.6–1.7 GB/s bus bandwidth, including
within a four-GPU group. This is not evidence that the copy-engine bridge plan
is faster for all-reduce; the target workload must be benchmarked separately.
