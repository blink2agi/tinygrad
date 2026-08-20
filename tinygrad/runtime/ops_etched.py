"""Public Etched reference backend for Tinygrad.

`DEV=ETCHED` exercises Tinygrad's real lowering and runtime boundaries through
the pure-Python Etched driver. It does not claim Sohu binary compatibility.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any
import base64, binascii, pickle

from tinygrad.device import Allocator, Buffer, Compiled, CompileError, Program, TinyELF
from tinygrad.helpers import cpu_profile
from tinygrad.runtime.ops_python import PythonCompiler, PythonProgram, PythonRenderer
from tinygrad.runtime.support.etched import EtchedBuffer, EtchedFormatError, PublicExecutable, PythonEtchedDriver
from tinygrad.uop.ops import UOp


def _metadata_from_payload(payload:bytes, error_type:type[Exception]=EtchedFormatError) -> dict[str, Any]:
  try: uops = pickle.loads(payload)
  except Exception as exc: raise error_type(f"Etched reference payload is not a lowered UOp list: {exc}") from exc
  if not isinstance(uops, list) or any(not isinstance(uop, UOp) for uop in uops):
    raise error_type("Etched reference payload is not a lowered UOp list")
  return {"format": "etched-public-executable", "version": 1, "public_ir_version": 1, "uop_count": len(uops),
          "op_histogram": dict(Counter(uop.op.name for uop in uops))}


class EtchedCompiler(PythonCompiler):
  def compile(self, src:str) -> bytes:
    try: payload = base64.b64decode(src, validate=True)
    except (ValueError, binascii.Error) as exc: raise CompileError(f"Etched renderer output is not valid base64: {exc}") from exc
    return PublicExecutable(_metadata_from_payload(payload, CompileError), payload).encode()


class EtchedRenderer(PythonRenderer):
  compiler = EtchedCompiler()


class EtchedAllocator(Allocator['EtchedDevice']):
  def _alloc(self, size, options): return self.dev.driver.allocate(size)
  def _free(self, opaque, options): self.dev.driver.free(opaque)
  def _as_buffer(self, src:EtchedBuffer) -> memoryview: return self.dev.driver.as_memoryview(src)
  def _copyin(self, dest:EtchedBuffer, src:memoryview):
    with cpu_profile("TINY -> ETCHED", f"{self.dev.device}:COPY"): self.dev.driver.copyin(dest, src)
  def _copyout(self, dest:memoryview, src:EtchedBuffer):
    with cpu_profile("ETCHED -> TINY", f"{self.dev.device}:COPY"): self.dev.driver.copyout(dest, src)
  def map(self, buf:Buffer): return buf.as_memoryview(force_zero_copy=True)
  def _offset(self, buf:EtchedBuffer, size:int, offset:int): return self.dev.driver.view(buf, offset, size)


class EtchedProgram(Program['EtchedDevice']):
  def __init__(self, dev:'EtchedDevice', obj:TinyELF):
    self.dev, self.executable = dev, PublicExecutable.decode(obj.lib)
    if self.executable.metadata != _metadata_from_payload(self.executable.payload):
      raise EtchedFormatError("Etched executable metadata does not match its payload")
    self.emulator = PythonProgram(dev, replace(obj, lib=self.executable.payload))  # type: ignore[arg-type]

  def __call__(self, *bufs:EtchedBuffer, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]|None=(1,1,1),
               vals:tuple[int, ...]=(), wait=False, **kw) -> float|None:
    views = tuple(self.dev.driver.as_memoryview(buf) for buf in bufs)
    def execute(): return self.emulator(*views, global_size=global_size, local_size=local_size or (1,1,1), vals=vals, wait=True, **kw)
    result = self.dev.driver.submit(self.executable, execute).result
    if not isinstance(result, float): raise RuntimeError("Etched reference executor did not return a duration")
    return result if wait else None


class EtchedDevice(Compiled):
  def __init__(self, device:str):
    self.driver = PythonEtchedDriver()
    super().__init__(device, EtchedAllocator(self), [EtchedRenderer], EtchedProgram)

  def synchronize(self): return self.driver.synchronize()
