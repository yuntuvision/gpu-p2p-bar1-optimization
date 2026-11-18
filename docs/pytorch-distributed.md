# PyTorch multi-GPU runbook

This runbook is sanitized: it contains no credentials or public SSH endpoint.
It describes the tested environment for two 8× RTX 4090 nodes.

## Installed environment

On both nodes:

```text
/home/user/venvs/torch-2.11-cu128
PyTorch 2.11.0+cu128
CUDA 12.8
NCCL 2.28.9
```

The user-level environment file is `/home/user/torch-env.sh`. It selects the
100GbE interface and currently disables NCCL IB because the kernel cannot load
`nvidia-peermem` and the non-GDR IB path benchmarked slower than TCP.

Node-specific interface values:

```text
Node A: enp195s0np0, 10.0.10.13
Node B: enp194s0np0, 10.0.10.14
```

Load the environment before using PyTorch:

```bash
source ~/torch-env.sh
```

## Launch commands

Single-node 8-GPU training:

```bash
source ~/torch-env.sh
torchrun --standalone --nproc_per_node=8 train.py
```

Two-node 16-GPU training, Node A:

```bash
source ~/torch-env.sh
torchrun --nnodes=2 --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=10.0.10.13 \
  --master_port=29501 \
  train.py
```

Two-node 16-GPU training, Node B:

```bash
source ~/torch-env.sh
torchrun --nnodes=2 --nproc_per_node=8 \
  --node_rank=1 \
  --master_addr=10.0.10.13 \
  --master_port=29501 \
  train.py
```

Use a different free `--master_port` for concurrent jobs. Start both commands
with the same training configuration and world size.

## Monitoring

```bash
watch -n 1 nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
free -h
df -h
```

## Validation

The following tests passed:

- Single-node 4-GPU and 8-GPU NCCL all-reduce correctness.
- Two-node 16-GPU NCCL all-reduce correctness.
- Single-node 4-GPU DDP forward/backward/optimizer steps.
- Two-node 16-GPU DDP forward/backward/optimizer steps; parameter checksum
  spread was zero.

For a 256 MiB two-node all-reduce, the measured effective one-way bandwidth was
approximately 0.175 GB/s with the non-GDR RoCE/IB path and 0.775 GB/s with TCP
over the 100GbE interfaces. These are workload-level all-reduce measurements,
not raw NIC line-rate measurements.

## Remaining bottlenecks

- `sudo modprobe nvidia-peermem` fails on both nodes with `Invalid argument`.
- The Ubuntu kernel does not export the peer-memory API required by the module;
  NCCL reports `GPU Direct RDMA Disabled`.
- [NVIDIA's driver guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/optional-components.html)
  documents that GPUDirect RDMA requires peer-memory kernel support or MLNX_OFED;
  replacing the kernel/RDMA stack is a maintenance-window task.
- Node B GPU1 is PCIe 4.0 x8; the other GPUs are x16.
- NumPy is not part of the minimal PyTorch environment, so PyTorch emits a
  non-fatal NumPy warning. Install it separately if the training stack needs it.

The temporary smoke-test scripts used during validation are not required for
normal training and should not be treated as a production launcher.
