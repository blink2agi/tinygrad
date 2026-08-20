# Etched public reference backend

This maintained Tinygrad fork adds a working `DEV=ETCHED` compiler/runtime path and a 100% Python userspace driver. It is a clean-room implementation built from public information for the [SemiAnalysis Etched/Tinygrad bounty](https://x.com/SemiAnalysis_/status/2090287730605355069).

The portable reference path is complete and tested. This project does not claim physical A0 execution: Etched has not publicly released the Sohu command opcodes, payload schema, memory registration UAPI, firmware handshake, executable binary format, or completion semantics, and no A0 hardware was available for testing.

## Quick start

From this repository:

```sh
DEV=ETCHED python3 examples/etched_demo.py
```

The demo performs a rectangular matrix multiplication through Tinygrad's normal scheduler, Etched renderer, executable envelope, allocator, Python submission queue, and completion path. It prints the result and queue evidence.

To run Tinygrad's smoke suite on the backend:

```sh
DEV=ETCHED python3 -m pytest -q test/test_tiny.py
```

## What is implemented

| Layer | Implementation | Status |
| --- | --- | --- |
| Tinygrad device | `tinygrad/runtime/ops_etched.py` | Working |
| Renderer/compiler | Lowered UOps in a versioned, hashed Etched public envelope | Working |
| Allocation | Aligned virtual addresses, ownership, bounded views, free validation | Working |
| Transfers | Host-to-device and device-to-host copies through the driver | Working |
| Queue | Ordered submissions, sequence IDs, completions, traces, failures, sync | Working |
| Reference device | Tinygrad's Python UOp semantics behind the Etched queue boundary | Working |
| Public semantic IR | Canonical patent-field matmul record | Working |
| PCI discovery | Linux sysfs identity `20a1:0001` | Working and fixture-tested |
| Linux command encoding | 64-byte `IORING_OP_URING_CMD` SQE matching Etched's public builder | Working and byte-tested |
| Raw Linux host queue | `io_uring_setup`/mmap/register/enter/CQE in 100% Python | Working; fake-kernel tested |
| Sohu A0 command submission | Vendor `cmd_op` and payload ABI | Awaiting Etched disclosure |

The default is the portable reference device. It does not report software execution as Sohu execution.

## Execution path

```text
Tensor operation
  -> Tinygrad schedule and lowering
  -> EtchedRenderer
  -> EtchedCompiler
  -> checked PublicExecutable
  -> EtchedProgram
  -> PythonEtchedDriver submission
  -> reference device execution
  -> completion and trace
```

`EtchedProgram` cannot call the emulator directly. Every launch is submitted to `PythonEtchedDriver`, which records a submission and a matching success or failure completion. Buffers are opaque `EtchedBuffer` records until the reference execution boundary.

## Public executable format

The reference executable starts with a fixed 52-byte little-endian header:

```text
magic[8] | version:u16 | flags:u16 | metadata_len:u32 | payload_len:u32 | sha256[32]
```

Canonical JSON metadata contains the format version, public-IR version, UOp count, and operation histogram. The SHA-256 digest covers the exact metadata and payload bytes. Decoding rejects wrong magic, unsupported versions or flags, invalid lengths, noncanonical JSON, schema mismatches, and digest corruption. Runtime construction also verifies that the metadata agrees with the decoded UOps.

This is a public reference format, not a guess at Etched's unpublished binary executable format.

## Public matmul IR

Etched's public patent figures name semantic fields for a matrix-multiplication instruction. `PublicMatmulIR` preserves those names in strict canonical JSON:

- preprocessing: `pre_bias_addr`, `pre_scale_addr`, `norm_on`;
- systolic array: `weight_addr`, `in_features`, `out_features`;
- postprocessing: `post_bias_addr`, `post_scale_addr`, `act_on`, `rope_on`, `glu_on`.

Addresses, dimensions, flags, section keys, versions, and canonical serialization are validated. The record is useful as a clean public compiler/driver handoff, but it is not presented as Sohu's binary encoding.

## Python userspace driver

`tinygrad/runtime/support/etched.py` contains the driver and uses only the Python standard library. It provides:

- monotonic 64-byte-aligned reference addresses;
- driver ownership and allocation identity checks;
- exact-size copies and bounds-checked nested views;
- base-only free and use-after-free rejection;
- serialized submissions and monotonically increasing sequence IDs;
- immutable submission, completion, and trace snapshots;
- exception-to-failed-completion propagation;
- deterministic synchronization;
- raw Linux `io_uring` setup, shared-ring mapping, fixed-file registration, SQE queueing, submission, and CQE decoding.

The software device is synchronous internally, which makes test results deterministic, while submission and completion remain separate records to preserve the physical-device boundary.

## Published Sohu/Linux boundary

Etched's public `io-uring` commit adds `SohuSendCmd` with these assignments:

- Linux opcode: `IORING_OP_URING_CMD` (`46` in the Linux UAPI used by Tinygrad);
- device descriptor: `fd`;
- vendor command: `cmd_op`;
- payload pointer: `addr`;
- payload length: `len`.

`SohuUringCommand.to_sqe()` encodes those fields into an otherwise-zero 64-byte Linux `io_uring_sqe`. Tests verify the fixed-file flag and the exact field offsets against Tinygrad's generated Linux definitions:

| Field | Byte offset | Encoding |
| --- | ---: | --- |
| `opcode` | 0 | unsigned 8-bit |
| fixed-file flag | 1 | unsigned 8-bit |
| `fd` | 4 | signed 32-bit |
| `cmd_op` | 8 | unsigned 32-bit |
| `addr` | 16 | unsigned 64-bit |
| `len` | 24 | unsigned 32-bit |
| `user_data` | 32 | unsigned 64-bit |

`LinuxIoUring` is the actual 100% Python host queue, not only an encoder. Using `ctypes`, `mmap`, and the Linux UAPI already carried by Tinygrad, it:

1. calls `io_uring_setup` and validates the kernel's 120-byte parameter block;
2. maps the submission ring, completion ring, and SQE array, including `IORING_FEAT_SINGLE_MMAP`;
3. registers the Sohu device descriptor as a fixed file;
4. publishes a 64-byte SQE, calls `io_uring_enter`, waits for a CQE, and checks `user_data` correlation;
5. returns signed device results and always closes mappings, registered files, and ring descriptors on success or failure.

The state machine is exercised against a fake kernel that reads the published SQE and writes a real-layout CQE. A Linux-only `IORING_OP_NOP` test is included for execution on a Linux kernel; this macOS development host cannot run that syscall.

`discover_sohu_devices()` scans `/sys/bus/pci/devices` and selects only the public Etched Sohu PCI identity `20a1:0001`. `LinuxSohuTransport` then validates Linux, PCI discovery, selection, and the device node. Its explicit `submit_raw(cmd_op, payload)` path opens the node, registers it in `LinuxIoUring`, pins the Python payload across DMA, submits `SohuSendCmd`, waits for completion, and converts negative CQE results to `OSError`.

The ordinary `submit()` path still fails closed instead of choosing a vendor command on the caller's behalf, because these public inputs are absent:

1. the valid Sohu `cmd_op` values;
2. the command payload schema and executable payload format;
3. memory registration, DMA mapping, and lifetime requirements;
4. the device-node/kernel-driver contract;
5. firmware compatibility and initialization handshake;
6. completion result and error semantics;
7. one A0 conformance vector.

Guessing any of these could DMA from the wrong address or submit a structurally valid command with destructive meaning. The ordinary path raises `EtchedPublicSpecIncomplete` until Etched supplies the contract. `submit_raw` accepts an explicit `cmd_op` and payload for an Etched engineer or hardware owner; the implementation assigns no guessed values.

## What Etched needs to provide for A0 bring-up

The remaining physical adapter is small. It needs:

```text
device node and minimum kernel version
cmd_op enum
payload structs with byte order, alignment, and version
buffer allocation/registration and IOMMU rules
firmware initialization/version query
completion status table
one input executable + buffers + expected completion/output
```

With those items, a typed `LinuxSohuTransport.submit()` can wrap the already-implemented `submit_raw` path and be tested against the conformance vector. No Tensor API, ring implementation, or compiler/runtime ownership boundary needs to change.

## Verification

Focused tests:

```sh
python3 -m pytest -q test/device/test_etched_driver.py test/device/test_etched.py test/device/test_etched_docs.py
```

Tinygrad smoke suite:

```sh
DEV=ETCHED python3 -m pytest -q test/test_tiny.py
```

Static checks:

```sh
ruff check tinygrad/runtime/ops_etched.py tinygrad/runtime/support/etched.py test/device/test_etched.py test/device/test_etched_driver.py
python3 -m compileall -q tinygrad/runtime/ops_etched.py tinygrad/runtime/support/etched.py examples/etched_demo.py
```

Bounty line-count proof for the backend plus driver:

```sh
wc -l tinygrad/runtime/ops_etched.py tinygrad/runtime/support/etched.py
```

The final submission report also counts all added Python, tests, examples, notices, and documentation so the ambiguous interpretation of the bounty's 10,000-line ceiling is covered conservatively.

## Public sources

- [SemiAnalysis bounty post](https://x.com/SemiAnalysis_/status/2090287730605355069)
- [Etched's `SohuSendCmd` commit (`aa7d43725fbf828d11fdab4356c8be90021cb6c9`)](https://github.com/etched-ai/io-uring/commit/aa7d43725fbf828d11fdab4356c8be90021cb6c9)
- [Etched public IR patent application US20250138820A1](https://patents.google.com/patent/US20250138820A1/en)
- [Etched public template patent application US20250156164A1](https://patents.google.com/patent/US20250156164A1/en)
- [Etched progress update describing Sohu A0](https://www.etched.com/progress/frontier-inference-clusters)
- [Tinygrad's generated Linux `io_uring` UAPI](../tinygrad/runtime/autogen/io_uring.py)
- [Public-source notice](../NOTICE.etched-public-sources.md)

## Security and trust

The reference payload uses Tinygrad's existing pickled Python UOps. Like Tinygrad's `PYTHON` backend, it must only compile and execute renderer output produced inside the trusted process. The SHA-256 digest detects accidental corruption; it is not a signature and does not make untrusted pickle safe.

No private Etched material is required. No vendor binary, kernel module, C extension, Rust library, or SDK is loaded.
