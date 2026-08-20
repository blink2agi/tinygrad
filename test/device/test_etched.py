import pickle, unittest

from tinygrad.device import ALL_DEVICES, Device, TinyELF
from tinygrad.helpers import Target
from tinygrad.runtime.ops_etched import EtchedCompiler, EtchedDevice, EtchedProgram, EtchedRenderer
from tinygrad.runtime.support.etched import EtchedFormatError, PublicExecutable
from tinygrad.uop.ops import Ops, UOp


def compiled_uops(uops:list[UOp]) -> bytes:
  renderer = EtchedRenderer(Target("ETCHED"))
  return EtchedCompiler().compile(renderer.render(uops))


class TestEtchedCompiler(unittest.TestCase):
  def test_wraps_lowered_uops_with_checked_metadata(self):
    uops = [UOp(Ops.NOOP), UOp(Ops.NOOP), UOp(Ops.SINK)]
    blob = compiled_uops(uops)
    executable = PublicExecutable.decode(blob)
    self.assertEqual(pickle.loads(executable.payload), uops)
    self.assertEqual(executable.metadata, {"format": "etched-public-executable", "op_histogram": {"NOOP": 2, "SINK": 1},
                                           "public_ir_version": 1, "uop_count": 3, "version": 1})
    self.assertEqual(blob, compiled_uops(uops))


class TestEtchedAllocator(unittest.TestCase):
  def test_delegates_alloc_copy_view_and_free(self):
    device = EtchedDevice("ETCHED")
    allocator = device.allocator
    buf = allocator.alloc(6)
    allocator._copyin(buf, memoryview(b"abcdef"))
    view = allocator._offset(buf, 3, 2)
    self.assertEqual(bytes(allocator._as_buffer(view)), b"cde")
    allocator._copyin(view, memoryview(b"XYZ"))
    result = bytearray(6)
    allocator._copyout(memoryview(result), buf)
    self.assertEqual(result, b"abXYZf")
    allocator.free(buf, 6)
    with self.assertRaisesRegex(RuntimeError, "freed"):
      allocator._as_buffer(buf)


class TestEtchedProgram(unittest.TestCase):
  def test_rejects_malformed_or_inconsistent_executable(self):
    device = EtchedDevice("ETCHED")
    with self.assertRaises(EtchedFormatError):
      EtchedProgram(device, TinyELF(b"not-an-executable", "bad", Target("ETCHED"), ()))

    valid = PublicExecutable.decode(compiled_uops([UOp(Ops.NOOP)]))
    inconsistent = PublicExecutable(valid.metadata | {"uop_count": 2}, valid.payload).encode()
    with self.assertRaisesRegex(EtchedFormatError, "metadata.*payload"):
      EtchedProgram(device, TinyELF(inconsistent, "bad-metadata", Target("ETCHED"), ()))

  def test_every_call_crosses_driver_submission_boundary(self):
    device = EtchedDevice("ETCHED")
    program = EtchedProgram(device, TinyELF(compiled_uops([UOp(Ops.NOOP)]), "noop", Target("ETCHED"), ()))
    self.assertIsNone(program(wait=False))
    waited = program(wait=True)
    self.assertIsInstance(waited, float)
    self.assertEqual([x.sequence for x in device.driver.completions], [1, 2])
    self.assertTrue(all(x.ok for x in device.driver.completions))
    self.assertEqual([x.kind for x in device.driver.trace], ["submit", "complete", "submit", "complete"])
    self.assertEqual(device.synchronize(), 2)


class TestEtchedDevice(unittest.TestCase):
  def test_device_is_discoverable_by_tinygrad(self):
    self.assertIn("ETCHED", ALL_DEVICES)
    self.assertIsInstance(Device["ETCHED"], EtchedDevice)
    self.assertEqual(Device["ETCHED"].renderer.target.device, "ETCHED")


if __name__ == "__main__": unittest.main()
