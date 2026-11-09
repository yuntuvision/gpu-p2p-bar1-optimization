#!/usr/bin/env python3
"""Measure one CUDA P2P pair, either one-way or simultaneous bidirectional."""

import ctypes
import statistics
import sys
import time


CUDA_SUCCESS = 0
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
BUFFER_BYTES = 256 * 1024 * 1024
WARMUP_ITERATIONS = 2
MEASURE_ITERATIONS = 20
ROUNDS = 5


def symbol(lib, *names):
    for name in names:
        try:
            return getattr(lib, name)
        except AttributeError:
            pass
    raise RuntimeError(f"missing CUDA symbol: {' or '.join(names)}")


def check(rc, operation):
    if rc != CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed with CUDA error {rc}")


if len(sys.argv) != 4 or sys.argv[1] not in {"uni", "bi"}:
    raise SystemExit(f"usage: {sys.argv[0]} uni|bi SRC DST")

mode = sys.argv[1]
src = int(sys.argv[2])
dst = int(sys.argv[3])
if src == dst or not (0 <= src < 8 and 0 <= dst < 8):
    raise SystemExit("SRC and DST must be distinct CUDA ordinals in [0, 7]")

lib = ctypes.CDLL("libcuda.so.1")
cuInit = symbol(lib, "cuInit")
cuDeviceGetCount = symbol(lib, "cuDeviceGetCount")
cuDeviceGet = symbol(lib, "cuDeviceGet")
cuDeviceCanAccessPeer = symbol(lib, "cuDeviceCanAccessPeer")
cuDevicePrimaryCtxRetain = symbol(lib, "cuDevicePrimaryCtxRetain")
cuDevicePrimaryCtxRelease = symbol(
    lib, "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease"
)
cuCtxSetCurrent = symbol(lib, "cuCtxSetCurrent")
cuCtxEnablePeerAccess = symbol(lib, "cuCtxEnablePeerAccess")
cuCtxSynchronize = symbol(lib, "cuCtxSynchronize")
cuMemAlloc = symbol(lib, "cuMemAlloc_v2", "cuMemAlloc")
cuMemFree = symbol(lib, "cuMemFree_v2", "cuMemFree")
cuMemcpyPeerAsync = symbol(lib, "cuMemcpyPeerAsync")
cuEventCreate = symbol(lib, "cuEventCreate")
cuEventRecord = symbol(lib, "cuEventRecord")
cuEventSynchronize = symbol(lib, "cuEventSynchronize")
cuEventElapsedTime = symbol(lib, "cuEventElapsedTime")
cuEventDestroy = symbol(lib, "cuEventDestroy_v2", "cuEventDestroy")
cuStreamCreate = symbol(lib, "cuStreamCreate")
cuStreamDestroy = symbol(lib, "cuStreamDestroy_v2", "cuStreamDestroy")

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
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
]
cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
cuEventSynchronize.argtypes = [ctypes.c_void_p]
cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
cuEventDestroy.argtypes = [ctypes.c_void_p]
cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
cuStreamDestroy.argtypes = [ctypes.c_void_p]


def enqueue_copy(src_index, dst_index, stream, iterations):
    for _ in range(iterations):
        check(
            cuMemcpyPeerAsync(
                buffers[dst_index],
                contexts[dst_index],
                buffers[src_index],
                contexts[src_index],
                BUFFER_BYTES,
                stream,
            ),
            f"cuMemcpyPeerAsync GPU{src_index}->GPU{dst_index}",
        )


check(cuInit(0), "cuInit")
count = ctypes.c_int()
check(cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
if count.value != 8:
    raise RuntimeError(f"expected 8 CUDA devices, found {count.value}")

devices = {}
contexts = {}
buffers = {}
streams = {}

try:
    for ordinal in (src, dst):
        device = ctypes.c_int()
        context = ctypes.c_void_p()
        ptr = ctypes.c_uint64()
        check(cuDeviceGet(ctypes.byref(device), ordinal), f"cuDeviceGet({ordinal})")
        check(
            cuDevicePrimaryCtxRetain(ctypes.byref(context), device.value),
            f"cuDevicePrimaryCtxRetain({ordinal})",
        )
        check(cuCtxSetCurrent(context), f"cuCtxSetCurrent({ordinal})")
        check(cuMemAlloc(ctypes.byref(ptr), BUFFER_BYTES), f"cuMemAlloc({ordinal})")
        stream = ctypes.c_void_p()
        check(cuStreamCreate(ctypes.byref(stream), 0), f"cuStreamCreate({ordinal})")
        devices[ordinal] = device.value
        contexts[ordinal] = context
        buffers[ordinal] = ptr.value
        streams[ordinal] = stream

    for owner, peer in ((src, dst), (dst, src)):
        can_access = ctypes.c_int()
        check(
            cuDeviceCanAccessPeer(
                ctypes.byref(can_access), devices[owner], devices[peer]
            ),
            f"cuDeviceCanAccessPeer({owner},{peer})",
        )
        if can_access.value != 1:
            raise RuntimeError(f"peer access unsupported: GPU{owner} -> GPU{peer}")
        check(cuCtxSetCurrent(contexts[owner]), f"set owner GPU{owner}")
        rc = cuCtxEnablePeerAccess(contexts[peer], 0)
        if rc not in (CUDA_SUCCESS, CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED):
            check(rc, f"enable GPU{owner} -> GPU{peer}")

    # Warm both directions so context setup and reciprocal mapping are excluded.
    for copy_src, copy_dst in ((src, dst), (dst, src)):
        check(cuCtxSetCurrent(contexts[copy_dst]), f"warm GPU{copy_src}->{copy_dst}")
        enqueue_copy(copy_src, copy_dst, streams[copy_dst], WARMUP_ITERATIONS)
        check(cuCtxSynchronize(), f"warm sync GPU{copy_src}->{copy_dst}")

    if mode == "uni":
        samples = []
        check(cuCtxSetCurrent(contexts[dst]), f"measure GPU{src}->{dst}")
        for _ in range(ROUNDS):
            start = ctypes.c_void_p()
            end = ctypes.c_void_p()
            check(cuEventCreate(ctypes.byref(start), 0), "cuEventCreate(start)")
            check(cuEventCreate(ctypes.byref(end), 0), "cuEventCreate(end)")
            try:
                check(cuEventRecord(start, streams[dst]), "cuEventRecord(start)")
                enqueue_copy(src, dst, streams[dst], MEASURE_ITERATIONS)
                check(cuEventRecord(end, streams[dst]), "cuEventRecord(end)")
                check(cuEventSynchronize(end), "cuEventSynchronize(end)")
                elapsed_ms = ctypes.c_float()
                check(
                    cuEventElapsedTime(ctypes.byref(elapsed_ms), start, end),
                    "cuEventElapsedTime",
                )
                transferred = BUFFER_BYTES * MEASURE_ITERATIONS
                samples.append(transferred / (elapsed_ms.value / 1000.0) / 1e9)
            finally:
                cuEventDestroy(start)
                cuEventDestroy(end)
        print(
            f"UNI src={src} dst={dst} median_GBps={statistics.median(samples):.3f} "
            f"min_GBps={min(samples):.3f} max_GBps={max(samples):.3f}"
        )
    else:
        samples = []
        for _ in range(ROUNDS):
            started = time.perf_counter()
            check(cuCtxSetCurrent(contexts[dst]), f"launch GPU{src}->{dst}")
            enqueue_copy(src, dst, streams[dst], MEASURE_ITERATIONS)
            check(cuCtxSetCurrent(contexts[src]), f"launch GPU{dst}->{src}")
            enqueue_copy(dst, src, streams[src], MEASURE_ITERATIONS)
            check(cuCtxSetCurrent(contexts[dst]), f"sync GPU{src}->{dst}")
            check(cuCtxSynchronize(), f"sync GPU{src}->{dst}")
            check(cuCtxSetCurrent(contexts[src]), f"sync GPU{dst}->{src}")
            check(cuCtxSynchronize(), f"sync GPU{dst}->{src}")
            seconds = time.perf_counter() - started
            transferred = 2 * BUFFER_BYTES * MEASURE_ITERATIONS
            samples.append(transferred / seconds / 1e9)
        print(
            f"BI pair={src}<->{dst} aggregate_median_GBps={statistics.median(samples):.3f} "
            f"min_GBps={min(samples):.3f} max_GBps={max(samples):.3f}"
        )
finally:
    for ordinal in (src, dst):
        context = contexts.get(ordinal)
        if not context:
            continue
        cuCtxSetCurrent(context)
        stream = streams.get(ordinal)
        if stream:
            cuStreamDestroy(stream)
        ptr = buffers.get(ordinal)
        if ptr:
            cuMemFree(ptr)
    cuCtxSetCurrent(None)
    for ordinal in (src, dst):
        device = devices.get(ordinal)
        if device is not None:
            cuDevicePrimaryCtxRelease(device)
