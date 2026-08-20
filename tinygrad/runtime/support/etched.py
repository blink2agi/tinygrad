"""Clean-room public Etched/Sohu host contracts.

This module intentionally separates the behavior that can be implemented from public
information from the unpublished Sohu silicon ABI. It contains no Etched SDK code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib, hmac, json, struct


PUBLIC_IR_VERSION = 1
PUBLIC_EXECUTABLE_VERSION = 1
PUBLIC_EXECUTABLE_MAGIC = b"SOHUPUB\0"
_EXECUTABLE_HEADER = struct.Struct("<8sHHII32s")
_MAX_METADATA_SIZE = 1 << 20
_MAX_PAYLOAD_SIZE = 1 << 30


class EtchedFormatError(ValueError):
  """Raised when a public Etched record is malformed or unsupported."""


def _canonical_json(value:Any) -> bytes:
  try: return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
  except (TypeError, ValueError) as exc: raise EtchedFormatError(f"value is not canonical JSON: {exc}") from exc


def _uint(name:str, value:Any, bits:int, *, nonzero:bool=False) -> int:
  if isinstance(value, bool) or not isinstance(value, int): raise TypeError(f"{name} must be an integer")
  minimum = 1 if nonzero else 0
  if not minimum <= value < 1 << bits: raise ValueError(f"{name} must be in [{minimum}, {(1 << bits) - 1}]")
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
    object.__setattr__(self, "metadata", _validated_metadata(self.metadata))
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
