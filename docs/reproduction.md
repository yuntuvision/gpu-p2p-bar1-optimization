# Reproduction guide

These commands assume a Linux host with the NVIDIA driver loaded, Python 3, and
no active GPU jobs. They are read-only from a system-configuration perspective,
but they allocate temporary device memory and perform CUDA copies.

## Preflight

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
nvidia-smi --query-gpu=index,name,pcie.link.gen.current,pcie.link.width.current --format=csv
```

Stop if the first command lists a job you do not own.

## One pair

```bash
python3 scripts/cuda-p2p-pair-benchmark.py uni 0 4
python3 scripts/cuda-p2p-pair-benchmark.py uni 4 0
python3 scripts/cuda-p2p-pair-benchmark.py bi 0 4
```

`bi` reports the aggregate of simultaneous opposite directions. It does not
claim that one direction alone is twice as fast.

## Four bridges

Run four `uni` processes concurrently for `0→4`, `1→5`, `2→6`, and `3→7`, then
repeat in reverse. Sum the per-process CUDA-event medians only after confirming
all four processes completed successfully.

## Relay probe

```bash
python3 scripts/cuda-p2p-relay-benchmark.py direct 1 0 4
python3 scripts/cuda-p2p-relay-benchmark.py relay 1 0 4
```

The relay script intentionally uses host synchronization after every hop. It is
a baseline/rejection test, not a production communication implementation.

## Postflight

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
journalctl -k --since '20 minutes ago' --no-pager | \
  grep -Ei 'NVRM: Xid|AER.*(fatal|uncorrected)|hardware error|machine check' || true
```

The expected result is no remaining test process and no new fatal error match.
