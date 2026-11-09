#!/usr/bin/env python3
"""Compare direct P2P with a correctness-preserving two-hop GPU relay."""

import ctypes
import sys
import time


SUCCESS = 0
ALREADY_ENABLED = 704
BYTES = 256 * 1024 * 1024
WARMUPS = 2
ITERS = 20


def sym(lib, *names):
    for name in names:
        try:
            return getattr(lib, name)
        except AttributeError:
            pass
    raise RuntimeError("missing CUDA symbol: " + ", ".join(names))


def ok(rc, what):
    if rc != SUCCESS:
        raise RuntimeError(f"{what} failed with CUDA error {rc}")


if len(sys.argv) != 5 or sys.argv[1] not in {"direct", "relay"}:
    raise SystemExit(f"usage: {sys.argv[0]} direct|relay SRC RELAY DST")
mode, src, relay, dst = sys.argv[1], *(int(x) for x in sys.argv[2:])
if len({src, relay, dst}) != 3:
    raise SystemExit("SRC, RELAY and DST must be distinct CUDA ordinals")

lib = ctypes.CDLL("libcuda.so.1")
cuInit = sym(lib, "cuInit")
cuDeviceGetCount = sym(lib, "cuDeviceGetCount")
cuDeviceGet = sym(lib, "cuDeviceGet")
cuDeviceCanAccessPeer = sym(lib, "cuDeviceCanAccessPeer")
cuDevicePrimaryCtxRetain = sym(lib, "cuDevicePrimaryCtxRetain")
cuDevicePrimaryCtxRelease = sym(lib, "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease")
cuCtxSetCurrent = sym(lib, "cuCtxSetCurrent")
cuCtxEnablePeerAccess = sym(lib, "cuCtxEnablePeerAccess")
cuCtxSynchronize = sym(lib, "cuCtxSynchronize")
cuMemAlloc = sym(lib, "cuMemAlloc_v2", "cuMemAlloc")
cuMemFree = sym(lib, "cuMemFree_v2", "cuMemFree")
cuMemcpyPeerAsync = sym(lib, "cuMemcpyPeerAsync")

cuInit.argtypes = [ctypes.c_uint]
cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cuDeviceCanAccessPeer.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
cuCtxEnablePeerAccess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
cuMemFree.argtypes = [ctypes.c_uint64]
cuMemcpyPeerAsync.argtypes = [
    ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_void_p,
]

ok(cuInit(0), "cuInit")
count = ctypes.c_int()
ok(cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
if count.value != 8:
    raise RuntimeError(f"expected 8 GPUs, found {count.value}")

ordinals = (src, relay, dst)
devices = {}
contexts = {}
buffers = {}


def enable(owner, peer):
    capable = ctypes.c_int()
    ok(cuDeviceCanAccessPeer(ctypes.byref(capable), devices[owner], devices[peer]),
       f"canAccessPeer({owner},{peer})")
    if capable.value != 1:
        raise RuntimeError(f"peer access unsupported: GPU{owner} -> GPU{peer}")
    ok(cuCtxSetCurrent(contexts[owner]), f"set GPU{owner}")
    rc = cuCtxEnablePeerAccess(contexts[peer], 0)
    if rc not in (SUCCESS, ALREADY_ENABLED):
        ok(rc, f"enable GPU{owner} -> GPU{peer}")


def copy(src_idx, dst_idx):
    ok(cuCtxSetCurrent(contexts[dst_idx]), f"set destination GPU{dst_idx}")
    ok(cuMemcpyPeerAsync(
        buffers[dst_idx], contexts[dst_idx], buffers[src_idx], contexts[src_idx],
        BYTES, None,
    ), f"copy GPU{src_idx} -> GPU{dst_idx}")
    ok(cuCtxSynchronize(), f"sync GPU{src_idx} -> GPU{dst_idx}")


try:
    for ordinal in ordinals:
        dev = ctypes.c_int()
        ctx = ctypes.c_void_p()
        ptr = ctypes.c_uint64()
        ok(cuDeviceGet(ctypes.byref(dev), ordinal), f"cuDeviceGet({ordinal})")
        ok(cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev.value),
           f"retain context {ordinal}")
        ok(cuCtxSetCurrent(ctx), f"set GPU{ordinal}")
        ok(cuMemAlloc(ctypes.byref(ptr), BYTES), f"allocate GPU{ordinal}")
        devices[ordinal] = dev.value
        contexts[ordinal] = ctx
        buffers[ordinal] = ptr.value

    if mode == "direct":
        enable(dst, src)
        for _ in range(WARMUPS):
            copy(src, dst)
        started = time.perf_counter()
        for _ in range(ITERS):
            copy(src, dst)
        elapsed = time.perf_counter() - started
        print(f"DIRECT src={src} dst={dst} GBps={BYTES * ITERS / elapsed / 1e9:.3f}")
    else:
        # The relay context owns the first copy; the destination context owns the
        # second. Host synchronization preserves correctness for this baseline.
        enable(relay, src)
        enable(dst, relay)
        for _ in range(WARMUPS):
            copy(src, relay)
            copy(relay, dst)
        started = time.perf_counter()
        for _ in range(ITERS):
            copy(src, relay)
            copy(relay, dst)
        elapsed = time.perf_counter() - started
        print(
            f"RELAY src={src} via={relay} dst={dst} "
            f"effective_GBps={BYTES * ITERS / elapsed / 1e9:.3f}"
        )
finally:
    for ordinal in ordinals:
        ctx = contexts.get(ordinal)
        if ctx is None:
            continue
        cuCtxSetCurrent(ctx)
        ptr = buffers.get(ordinal)
        if ptr:
            cuMemFree(ptr)
    cuCtxSetCurrent(None)
    for ordinal in ordinals:
        dev = devices.get(ordinal)
        if dev is not None:
            cuDevicePrimaryCtxRelease(dev)
