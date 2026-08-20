"""Clean-room public Etched/Sohu host contracts.

This module intentionally separates the behavior that can be implemented from public
information from the unpublished Sohu silicon ABI. It contains no Etched SDK code.
"""
from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping
import hashlib, hmac, itertools, json, struct, sys, threading


PUBLIC_IR_VERSION = 1
PUBLIC_EXECUTABLE_VERSION = 1
PUBLIC_EXECUTABLE_MAGIC = b"SOHUPUB\0"
_EXECUTABLE_HEADER = struct.Struct("<8sHHII32s")
_MAX_METADATA_SIZE = 1 << 20
_MAX_PAYLOAD_SIZE = 1 << 30


class EtchedFormatError(ValueError):
  """Raised when a public Etched record is malformed or unsupported."""


class EtchedHardwareUnavailable(RuntimeError):
  """Raised when the requested physical Sohu transport cannot be opened."""


class EtchedPublicSpecIncomplete(RuntimeError):
  """Raised instead of guessing a silicon ABI that Etched has not published."""


def _plain_json(value:Any) -> Any:
  if isinstance(value, ABCMapping): return {key:_plain_json(item) for key,item in value.items()}
  if isinstance(value, (list, tuple)): return [_plain_json(item) for item in value]
  return value


def _canonical_json(value:Any) -> bytes:
  try: return json.dumps(_plain_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
  except (TypeError, ValueError) as exc: raise EtchedFormatError(f"value is not canonical JSON: {exc}") from exc


def _uint(name:str, value:Any, bits:int, *, nonzero:bool=False) -> int:
  if isinstance(value, bool) or not isinstance(value, int): raise TypeError(f"{name} must be an integer")
  minimum = 1 if nonzero else 0
  if not minimum <= value < 1 << bits: raise ValueError(f"{name} must be in [{minimum}, {(1 << bits) - 1}]")
  return value


def _sint(name:str, value:Any, bits:int) -> int:
  if isinstance(value, bool) or not isinstance(value, int): raise TypeError(f"{name} must be an integer")
  if not -(1 << (bits-1)) <= value < 1 << (bits-1):
    raise ValueError(f"{name} must fit in a signed {bits}-bit integer")
  return value


def _flag(name:str, value:Any) -> bool:
  if type(value) is not bool: raise TypeError(f"{name} must be a bool")
  return value


@dataclass(frozen=True)
class PublicMatmulIR:
  """Semantic matrix-multiplication record using Etched's published patent field names.

  This canonical JSON record is a public interoperability contract, not Sohu's
  undisclosed binary executable representation.
  """
  weight_addr:int
  in_features:int
  out_features:int
  pre_bias_addr:int = 0
  pre_scale_addr:int = 0
  norm_on:bool = False
  post_bias_addr:int = 0
  post_scale_addr:int = 0
  act_on:bool = False
  rope_on:bool = False
  glu_on:bool = False

  def __post_init__(self):
    for name in ("weight_addr", "pre_bias_addr", "pre_scale_addr", "post_bias_addr", "post_scale_addr"):
      _uint(name, getattr(self, name), 64)
    for name in ("in_features", "out_features"): _uint(name, getattr(self, name), 32, nonzero=True)
    for name in ("norm_on", "act_on", "rope_on", "glu_on"): _flag(name, getattr(self, name))

  def to_dict(self) -> dict[str, Any]:
    return {
      "format": "etched-public-matmul-ir",
      "version": PUBLIC_IR_VERSION,
      "preprocessing": {"pre_bias_addr": self.pre_bias_addr, "pre_scale_addr": self.pre_scale_addr, "norm_on": self.norm_on},
      "systolic_array": {"weight_addr": self.weight_addr, "in_features": self.in_features, "out_features": self.out_features},
      "postprocessing": {"post_bias_addr": self.post_bias_addr, "post_scale_addr": self.post_scale_addr, "act_on": self.act_on,
                         "rope_on": self.rope_on, "glu_on": self.glu_on},
    }

  def to_bytes(self) -> bytes: return _canonical_json(self.to_dict())

  @classmethod
  def from_bytes(cls, blob:bytes) -> PublicMatmulIR:
    if not isinstance(blob, bytes): raise TypeError("public matmul IR must be bytes")
    try: decoded = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise EtchedFormatError(f"invalid public matmul IR JSON: {exc}") from exc
    if _canonical_json(decoded) != blob: raise EtchedFormatError("public matmul IR JSON is not canonical")
    try:
      if set(decoded) != {"format", "version", "preprocessing", "systolic_array", "postprocessing"}: raise KeyError("top level")
      if decoded["format"] != "etched-public-matmul-ir" or decoded["version"] != PUBLIC_IR_VERSION: raise KeyError("identity")
      pre, systolic, post = decoded["preprocessing"], decoded["systolic_array"], decoded["postprocessing"]
      if set(pre) != {"pre_bias_addr", "pre_scale_addr", "norm_on"}: raise KeyError("preprocessing")
      if set(systolic) != {"weight_addr", "in_features", "out_features"}: raise KeyError("systolic_array")
      if set(post) != {"post_bias_addr", "post_scale_addr", "act_on", "rope_on", "glu_on"}: raise KeyError("postprocessing")
      return cls(weight_addr=systolic["weight_addr"], in_features=systolic["in_features"], out_features=systolic["out_features"],
                 pre_bias_addr=pre["pre_bias_addr"], pre_scale_addr=pre["pre_scale_addr"], norm_on=pre["norm_on"],
                 post_bias_addr=post["post_bias_addr"], post_scale_addr=post["post_scale_addr"], act_on=post["act_on"],
                 rope_on=post["rope_on"], glu_on=post["glu_on"])
    except (KeyError, TypeError, ValueError) as exc: raise EtchedFormatError(f"public matmul IR schema is invalid: {exc}") from exc


def _validated_metadata(metadata:Mapping[str, Any]) -> dict[str, Any]:
  if not isinstance(metadata, Mapping): raise TypeError("executable metadata must be a mapping")
  copied = json.loads(_canonical_json(metadata))
  required = {"format", "version", "public_ir_version", "uop_count", "op_histogram"}
  if set(copied) != required: raise EtchedFormatError("executable metadata schema has missing or extra fields")
  if copied["format"] != "etched-public-executable": raise EtchedFormatError("executable metadata format is invalid")
  if copied["version"] != PUBLIC_EXECUTABLE_VERSION: raise EtchedFormatError("executable metadata version is unsupported")
  if copied["public_ir_version"] != PUBLIC_IR_VERSION: raise EtchedFormatError("public IR version is unsupported")
  _uint("uop_count", copied["uop_count"], 32)
  histogram = copied["op_histogram"]
  if not isinstance(histogram, dict) or any(not isinstance(name, str) or not name for name in histogram):
    raise EtchedFormatError("op_histogram must map nonempty operation names to counts")
  for name, count in histogram.items(): _uint(f"op_histogram[{name!r}]", count, 32)
  return copied


@dataclass(frozen=True)
class PublicExecutable:
  """Versioned, integrity-checked envelope for the public reference executor."""
  metadata:Mapping[str, Any]
  payload:bytes

  def __post_init__(self):
    metadata = _validated_metadata(self.metadata)
    metadata["op_histogram"] = MappingProxyType(metadata["op_histogram"])
    object.__setattr__(self, "metadata", MappingProxyType(metadata))
    if not isinstance(self.payload, bytes): raise TypeError("executable payload must be bytes")
    if len(self.payload) > _MAX_PAYLOAD_SIZE: raise EtchedFormatError("executable payload is too large")

  def encode(self) -> bytes:
    metadata = _canonical_json(self.metadata)
    if len(metadata) > _MAX_METADATA_SIZE: raise EtchedFormatError("executable metadata is too large")
    digest = hashlib.sha256(metadata + self.payload).digest()
    header = _EXECUTABLE_HEADER.pack(PUBLIC_EXECUTABLE_MAGIC, PUBLIC_EXECUTABLE_VERSION, 0, len(metadata), len(self.payload), digest)
    return header + metadata + self.payload

  @classmethod
  def decode(cls, blob:bytes) -> PublicExecutable:
    if not isinstance(blob, bytes): raise TypeError("public executable must be bytes")
    if len(blob) < _EXECUTABLE_HEADER.size: raise EtchedFormatError("public executable length is shorter than its header")
    magic, version, flags, metadata_len, payload_len, digest = _EXECUTABLE_HEADER.unpack_from(blob)
    if magic != PUBLIC_EXECUTABLE_MAGIC: raise EtchedFormatError("public executable magic is invalid")
    if version != PUBLIC_EXECUTABLE_VERSION: raise EtchedFormatError(f"public executable version {version} is unsupported")
    if flags != 0: raise EtchedFormatError("public executable flags are unsupported")
    if metadata_len > _MAX_METADATA_SIZE or payload_len > _MAX_PAYLOAD_SIZE: raise EtchedFormatError("public executable length exceeds limits")
    expected_len = _EXECUTABLE_HEADER.size + metadata_len + payload_len
    if len(blob) != expected_len: raise EtchedFormatError(f"public executable length is {len(blob)}, expected {expected_len}")
    metadata_blob = blob[_EXECUTABLE_HEADER.size:_EXECUTABLE_HEADER.size+metadata_len]
    payload = blob[_EXECUTABLE_HEADER.size+metadata_len:]
    if not hmac.compare_digest(digest, hashlib.sha256(metadata_blob + payload).digest()):
      raise EtchedFormatError("public executable digest does not match")
    try: metadata = json.loads(metadata_blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise EtchedFormatError(f"invalid executable metadata JSON: {exc}") from exc
    if _canonical_json(metadata) != metadata_blob: raise EtchedFormatError("executable metadata JSON is not canonical")
    return cls(metadata, payload)


_DRIVER_IDS = itertools.count(1)
_REFERENCE_ADDRESS_BASE = 0x1000_0000_0000
_ALLOCATION_ALIGNMENT = 64


@dataclass(frozen=True)
class EtchedBuffer:
  """Opaque driver buffer or bounded view with a stable reference address."""
  owner:int
  allocation:int
  address:int
  size:int
  offset:int = 0


@dataclass
class _Allocation:
  address:int
  size:int
  storage:bytearray


@dataclass(frozen=True)
class EtchedSubmission:
  sequence:int
  executable_sha256:str
  payload_size:int


@dataclass(frozen=True)
class EtchedCompletion:
  sequence:int
  ok:bool
  result:Any = None
  error:str|None = None


@dataclass(frozen=True)
class EtchedTraceEvent:
  kind:str
  sequence:int
  detail:str


class PythonEtchedDriver:
  """100% Python allocation, submission, completion, and reference-device driver.

  Execution is serialized and synchronous, but submissions and completions are
  separately recorded to preserve a hardware-compatible ownership boundary.
  """
  def __init__(self):
    self._owner = next(_DRIVER_IDS)
    self._next_address, self._next_allocation, self._next_sequence = _REFERENCE_ADDRESS_BASE, 1, 1
    self._allocations:dict[int, _Allocation] = {}
    self._submissions:list[EtchedSubmission] = []
    self._completions:list[EtchedCompletion] = []
    self._trace:list[EtchedTraceEvent] = []
    self._pending:set[int] = set()
    self._lock = threading.RLock()

  @property
  def submissions(self) -> tuple[EtchedSubmission, ...]:
    with self._lock: return tuple(self._submissions)

  @property
  def completions(self) -> tuple[EtchedCompletion, ...]:
    with self._lock: return tuple(self._completions)

  @property
  def trace(self) -> tuple[EtchedTraceEvent, ...]:
    with self._lock: return tuple(self._trace)

  def allocate(self, size:int) -> EtchedBuffer:
    _uint("allocation size", size, 63, nonzero=True)
    with self._lock:
      address = (self._next_address + _ALLOCATION_ALIGNMENT - 1) & -_ALLOCATION_ALIGNMENT
      allocation = self._next_allocation
      self._next_allocation += 1
      self._next_address = address + ((size + _ALLOCATION_ALIGNMENT - 1) & -_ALLOCATION_ALIGNMENT)
      self._allocations[allocation] = _Allocation(address, size, bytearray(size))
      return EtchedBuffer(self._owner, allocation, address, size)

  def _resolve(self, buf:EtchedBuffer) -> tuple[_Allocation, int]:
    if not isinstance(buf, EtchedBuffer): raise TypeError("expected an EtchedBuffer")
    if buf.owner != self._owner: raise ValueError("buffer belongs to a different driver")
    if (allocation:=self._allocations.get(buf.allocation)) is None: raise RuntimeError("buffer allocation has been freed")
    if buf.offset < 0 or buf.size <= 0 or buf.offset + buf.size > allocation.size or buf.address != allocation.address + buf.offset:
      raise ValueError("buffer descriptor is outside its allocation")
    return allocation, buf.offset

  def view(self, buf:EtchedBuffer, offset:int, size:int) -> EtchedBuffer:
    allocation, base_offset = self._resolve(buf)
    _uint("view offset", offset, 63)
    _uint("view size", size, 63, nonzero=True)
    if offset + size > buf.size: raise ValueError("view is outside its source buffer")
    absolute_offset = base_offset + offset
    return EtchedBuffer(self._owner, buf.allocation, allocation.address + absolute_offset, size, absolute_offset)

  def as_memoryview(self, buf:EtchedBuffer) -> memoryview:
    with self._lock:
      allocation, offset = self._resolve(buf)
      return memoryview(allocation.storage)[offset:offset+buf.size]

  def copyin(self, dest:EtchedBuffer, src:memoryview) -> None:
    if not isinstance(src, memoryview): raise TypeError("copy source must be a memoryview")
    if src.nbytes != dest.size: raise ValueError(f"copy size {src.nbytes} does not match buffer size {dest.size}")
    self.as_memoryview(dest)[:] = src.cast("B")

  def copyout(self, dest:memoryview, src:EtchedBuffer) -> None:
    if not isinstance(dest, memoryview): raise TypeError("copy destination must be a memoryview")
    if dest.nbytes != src.size: raise ValueError(f"copy size {dest.nbytes} does not match buffer size {src.size}")
    dest.cast("B")[:] = self.as_memoryview(src)

  def free(self, buf:EtchedBuffer) -> None:
    with self._lock:
      allocation, _ = self._resolve(buf)
      if buf.offset != 0 or buf.address != allocation.address or buf.size != allocation.size:
        raise ValueError("only a base allocation can be freed")
      del self._allocations[buf.allocation]

  def submit(self, executable:PublicExecutable, execute:Callable[[], Any]) -> EtchedCompletion:
    if not isinstance(executable, PublicExecutable): raise TypeError("submission requires a PublicExecutable")
    if not callable(execute): raise TypeError("submission executor must be callable")
    with self._lock:
      sequence = self._next_sequence
      self._next_sequence += 1
      digest = hashlib.sha256(executable.encode()).hexdigest()
      self._submissions.append(EtchedSubmission(sequence, digest, len(executable.payload)))
      self._trace.append(EtchedTraceEvent("submit", sequence, digest))
      self._pending.add(sequence)
      try:
        result = execute()
      except Exception as exc:
        completion = EtchedCompletion(sequence, False, error=f"{type(exc).__name__}: {exc}")
        self._completions.append(completion)
        self._trace.append(EtchedTraceEvent("error", sequence, completion.error or ""))
        self._pending.remove(sequence)
        raise
      completion = EtchedCompletion(sequence, True, result=result)
      self._completions.append(completion)
      self._trace.append(EtchedTraceEvent("complete", sequence, digest))
      self._pending.remove(sequence)
      return completion

  def synchronize(self) -> int:
    with self._lock:
      if self._pending: raise RuntimeError(f"reference driver has pending submissions: {sorted(self._pending)}")
      return self._completions[-1].sequence if self._completions else 0


SOHU_PCI_VENDOR_ID = 0x20A1
SOHU_PCI_DEVICE_ID = 0x0001
IORING_OP_URING_CMD = 46
IOSQE_FIXED_FILE = 1


@dataclass(frozen=True)
class SohuPciDevice:
  bdf:str
  sysfs_path:Path
  vendor_id:int = SOHU_PCI_VENDOR_ID
  device_id:int = SOHU_PCI_DEVICE_ID


def discover_sohu_devices(sysfs_root:Path|str=Path("/sys/bus/pci/devices")) -> tuple[SohuPciDevice, ...]:
  """Discover public PCI identity 20a1:0001 without requiring a vendor library."""
  root = Path(sysfs_root)
  try: candidates = sorted(root.iterdir(), key=lambda path:path.name)
  except OSError: return ()
  found:list[SohuPciDevice] = []
  for path in candidates:
    try: vendor, device = int((path / "vendor").read_text().strip(), 0), int((path / "device").read_text().strip(), 0)
    except (OSError, ValueError): continue
    if (vendor, device) == (SOHU_PCI_VENDOR_ID, SOHU_PCI_DEVICE_ID): found.append(SohuPciDevice(path.name, path))
  return tuple(found)


@dataclass(frozen=True)
class SohuUringCommand:
  """Etched's publicly committed `SohuSendCmd` mapped onto Linux `io_uring_sqe`.

  Field offsets follow Linux's 64-byte SQE. `fixed_file=True` models the fixed
  descriptor form accepted by the public Rust builder.
  """
  fd:int
  cmd_op:int
  addr:int
  length:int
  user_data:int = 0
  fixed_file:bool = True

  def __post_init__(self):
    _sint("fd", self.fd, 32)
    _uint("cmd_op", self.cmd_op, 32)
    _uint("addr", self.addr, 64)
    _uint("length", self.length, 32, nonzero=True)
    _uint("user_data", self.user_data, 64)
    _flag("fixed_file", self.fixed_file)

  def to_sqe(self) -> bytes:
    sqe = bytearray(64)
    struct.pack_into("<B", sqe, 0, IORING_OP_URING_CMD)
    struct.pack_into("<B", sqe, 1, IOSQE_FIXED_FILE if self.fixed_file else 0)
    struct.pack_into("<i", sqe, 4, self.fd)
    struct.pack_into("<I", sqe, 8, self.cmd_op)
    struct.pack_into("<Q", sqe, 16, self.addr)
    struct.pack_into("<I", sqe, 24, self.length)
    struct.pack_into("<Q", sqe, 32, self.user_data)
    return bytes(sqe)


class LinuxSohuTransport:
  """Fail-closed physical transport boundary based only on public information.

  Construction proves the host and PCI identity are plausible. Submission is
  deliberately blocked until Etched publishes the device command contract.
  """
  def __init__(self, device_node:Path|str, *, sysfs_root:Path|str=Path("/sys/bus/pci/devices"),
               platform:str|None=None, bdf:str|None=None):
    active_platform = sys.platform if platform is None else platform
    if not active_platform.startswith("linux"): raise EtchedHardwareUnavailable("physical Sohu transport requires Linux")
    devices = discover_sohu_devices(sysfs_root)
    if not devices: raise EtchedHardwareUnavailable("no Etched Sohu PCI device 20a1:0001 was found")
    matching = [device for device in devices if bdf is None or device.bdf == bdf]
    if not matching: raise EtchedHardwareUnavailable(f"Sohu PCI device {bdf!r} was not found")
    self.device = matching[0]
    self.device_node = Path(device_node)
    if not self.device_node.exists() or self.device_node.is_dir():
      raise EtchedHardwareUnavailable(f"Sohu device node {self.device_node} does not exist")

  def submit(self, payload:bytes):
    if not isinstance(payload, bytes): raise TypeError("Sohu command payload must be bytes")
    if not payload: raise ValueError("Sohu command payload must be nonempty")
    raise EtchedPublicSpecIncomplete("Etched has not published Sohu cmd_op values, payload schema, memory registration UAPI, "
                                     "firmware handshake, or completion semantics; refusing to guess the physical ABI")
