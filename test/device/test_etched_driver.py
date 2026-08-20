import hashlib, json, struct, unittest

from tinygrad.runtime.support.etched import (
  EtchedFormatError, PublicExecutable, PublicMatmulIR, PythonEtchedDriver,
)


class TestPublicMatmulIR(unittest.TestCase):
  def test_canonical_patent_field_encoding(self):
    ir = PublicMatmulIR(weight_addr=0x1000, in_features=16, out_features=32, pre_bias_addr=0x2000, pre_scale_addr=0x3000,
                        norm_on=True, post_bias_addr=0x4000, post_scale_addr=0x5000, act_on=True, rope_on=False, glu_on=True)
    expected = {
      "format": "etched-public-matmul-ir",
      "postprocessing": {"act_on": True, "glu_on": True, "post_bias_addr": 0x4000, "post_scale_addr": 0x5000, "rope_on": False},
      "preprocessing": {"norm_on": True, "pre_bias_addr": 0x2000, "pre_scale_addr": 0x3000},
      "systolic_array": {"in_features": 16, "out_features": 32, "weight_addr": 0x1000},
      "version": 1,
    }
    self.assertEqual(ir.to_bytes(), json.dumps(expected, sort_keys=True, separators=(",", ":")).encode())
    self.assertEqual(PublicMatmulIR.from_bytes(ir.to_bytes()), ir)

  def test_rejects_invalid_addresses_dimensions_and_flags(self):
    valid = {"weight_addr": 0x1000, "in_features": 16, "out_features": 32}
    for field, value in [("weight_addr", -1), ("weight_addr", 1 << 64), ("weight_addr", True),
                         ("in_features", 0), ("in_features", 1 << 32), ("in_features", True), ("act_on", 1)]:
      with self.subTest(field=field, value=value), self.assertRaises((TypeError, ValueError)):
        PublicMatmulIR(**(valid | {field: value}))

  def test_rejects_noncanonical_or_unknown_json(self):
    ir = PublicMatmulIR(weight_addr=1, in_features=2, out_features=3)
    spaced = json.dumps(json.loads(ir.to_bytes())).encode()
    with self.assertRaisesRegex(EtchedFormatError, "canonical"):
      PublicMatmulIR.from_bytes(spaced)

    decoded = json.loads(ir.to_bytes())
    decoded["surprise"] = 1
    with self.assertRaisesRegex(EtchedFormatError, "schema"):
      PublicMatmulIR.from_bytes(json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode())


class TestPublicExecutable(unittest.TestCase):
  metadata = {"format": "etched-public-executable", "op_histogram": {"ADD": 1, "STORE": 2}, "public_ir_version": 1,
              "uop_count": 3, "version": 1}

  def test_deterministic_checked_round_trip(self):
    executable = PublicExecutable(self.metadata, b"lowered-uops")
    blob = executable.encode()
    self.assertEqual(blob, executable.encode())
    self.assertEqual(PublicExecutable.decode(blob), executable)
    self.assertEqual(len(blob), 52 + len(json.dumps(self.metadata, sort_keys=True, separators=(",", ":")).encode()) + len(executable.payload))

  def test_rejects_invalid_metadata(self):
    for metadata in [self.metadata | {"version": 2}, self.metadata | {"uop_count": -1}, self.metadata | {"uop_count": True},
                     self.metadata | {"op_histogram": {"ADD": -1}}, self.metadata | {"extra": 1}]:
      with self.subTest(metadata=metadata), self.assertRaises((EtchedFormatError, TypeError, ValueError)):
        PublicExecutable(metadata, b"payload")

  def test_rejects_bad_magic_version_truncation_and_digest(self):
    blob = bytearray(PublicExecutable(self.metadata, b"payload").encode())
    cases = []
    bad_magic = blob.copy()
    bad_magic[0] ^= 0xFF
    cases.append((bad_magic, "magic"))
    bad_version = blob.copy()
    struct.pack_into("<H", bad_version, 8, 2)
    cases.append((bad_version, "version"))
    cases.append((blob[:-1], "length"))
    bad_digest = blob.copy()
    bad_digest[-1] ^= 0xFF
    cases.append((bad_digest, "digest"))
    for damaged, message in cases:
      with self.subTest(message=message), self.assertRaisesRegex(EtchedFormatError, message):
        PublicExecutable.decode(bytes(damaged))

  def test_rejects_noncanonical_metadata_even_with_valid_digest(self):
    metadata = json.dumps(self.metadata, sort_keys=True).encode()
    payload = b"payload"
    header = struct.pack("<8sHHII32s", b"SOHUPUB\0", 1, 0, len(metadata), len(payload), hashlib.sha256(metadata + payload).digest())
    blob = header + metadata + payload
    with self.assertRaisesRegex(EtchedFormatError, "canonical"):
      PublicExecutable.decode(blob)


def reference_executable(payload:bytes=b"lowered-uops") -> PublicExecutable:
  return PublicExecutable({"format": "etched-public-executable", "op_histogram": {"STORE": 1}, "public_ir_version": 1,
                           "uop_count": 1, "version": 1}, payload)


class TestPythonEtchedMemory(unittest.TestCase):
  def test_aligned_monotonic_allocations_and_copies(self):
    driver = PythonEtchedDriver()
    first, second = driver.allocate(3), driver.allocate(5)
    self.assertEqual(first.address % 64, 0)
    self.assertEqual(second.address % 64, 0)
    self.assertGreaterEqual(second.address - first.address, 64)

    driver.copyin(first, memoryview(b"abc"))
    result = bytearray(3)
    driver.copyout(memoryview(result), first)
    self.assertEqual(result, b"abc")

  def test_bounded_views_share_the_allocation(self):
    driver = PythonEtchedDriver()
    base = driver.allocate(6)
    driver.copyin(base, memoryview(b"abcdef"))
    view = driver.view(base, 2, 3)
    self.assertEqual((view.address, view.size), (base.address + 2, 3))
    driver.copyin(view, memoryview(b"XYZ"))
    self.assertEqual(bytes(driver.as_memoryview(base)), b"abXYZf")

    for offset, size in [(-1, 1), (0, 0), (6, 1), (4, 3)]:
      with self.subTest(offset=offset, size=size), self.assertRaises(ValueError):
        driver.view(base, offset, size)

  def test_rejects_invalid_size_copy_and_ownership(self):
    driver, other = PythonEtchedDriver(), PythonEtchedDriver()
    with self.assertRaises(ValueError): driver.allocate(0)
    buf = driver.allocate(3)
    with self.assertRaisesRegex(ValueError, "copy size"):
      driver.copyin(buf, memoryview(b"xx"))
    with self.assertRaisesRegex(ValueError, "copy size"):
      driver.copyout(memoryview(bytearray(2)), buf)
    with self.assertRaisesRegex(ValueError, "different driver"):
      other.as_memoryview(buf)

  def test_free_rejects_views_and_use_after_free(self):
    driver = PythonEtchedDriver()
    buf = driver.allocate(4)
    with self.assertRaisesRegex(ValueError, "base allocation"):
      driver.free(driver.view(buf, 1, 2))
    driver.free(buf)
    with self.assertRaisesRegex(RuntimeError, "freed"):
      driver.as_memoryview(buf)
    with self.assertRaisesRegex(RuntimeError, "freed"):
      driver.free(buf)


class TestPythonEtchedQueue(unittest.TestCase):
  def test_submission_completion_order_and_trace(self):
    driver = PythonEtchedDriver()
    first = driver.submit(reference_executable(b"one"), lambda: "first-result")
    second = driver.submit(reference_executable(b"two"), lambda: 22)
    self.assertEqual((first.sequence, first.ok, first.result, first.error), (1, True, "first-result", None))
    self.assertEqual((second.sequence, second.ok, second.result, second.error), (2, True, 22, None))
    self.assertEqual([x.sequence for x in driver.submissions], [1, 2])
    self.assertEqual([x.sequence for x in driver.completions], [1, 2])
    self.assertEqual([(x.kind, x.sequence) for x in driver.trace], [("submit", 1), ("complete", 1), ("submit", 2), ("complete", 2)])
    self.assertEqual(driver.synchronize(), 2)
    self.assertIsInstance(driver.submissions, tuple)

  def test_failure_is_completed_and_reraised(self):
    driver = PythonEtchedDriver()

    def fail(): raise RuntimeError("device exploded")

    with self.assertRaisesRegex(RuntimeError, "device exploded"):
      driver.submit(reference_executable(), fail)
    self.assertEqual(len(driver.completions), 1)
    self.assertFalse(driver.completions[0].ok)
    self.assertEqual(driver.completions[0].error, "RuntimeError: device exploded")
    self.assertEqual([(x.kind, x.sequence) for x in driver.trace], [("submit", 1), ("error", 1)])
    self.assertEqual(driver.synchronize(), 1)

  def test_rejects_non_executable_and_non_callable(self):
    driver = PythonEtchedDriver()
    with self.assertRaises(TypeError): driver.submit(b"not-an-executable", lambda: None)  # type: ignore[arg-type]
    with self.assertRaises(TypeError): driver.submit(reference_executable(), None)  # type: ignore[arg-type]


if __name__ == "__main__": unittest.main()
