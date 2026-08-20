import hashlib, json, os, struct, sys, tempfile, unittest
from pathlib import Path

from tinygrad.runtime.support.etched import (
  EtchedFormatError, EtchedHardwareUnavailable, EtchedPublicSpecIncomplete, IoUringError, IoUringQueueFull, LinuxIoUring,
  LinuxSohuTransport, PublicExecutable, PublicMatmulIR, PythonEtchedDriver, SohuUringCommand, discover_sohu_devices,
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

  def test_metadata_is_deeply_immutable(self):
    executable = PublicExecutable(self.metadata, b"payload")
    with self.assertRaises(TypeError): executable.metadata["uop_count"] = 4  # type: ignore[index]
    with self.assertRaises(TypeError): executable.metadata["op_histogram"]["ADD"] = 9  # type: ignore[index]

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


class TestSohuDiscovery(unittest.TestCase):
  @staticmethod
  def add_pci_device(root:Path, bdf:str, vendor:str, device:str):
    path = root / bdf
    path.mkdir()
    (path / "vendor").write_text(vendor)
    (path / "device").write_text(device)

  def test_finds_only_sohu_and_sorts_by_bdf(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      self.add_pci_device(root, "0000:81:00.0", "0x20A1\n", "0x0001\n")
      self.add_pci_device(root, "0000:03:00.0", "0x20a1", "0x0001")
      self.add_pci_device(root, "0000:02:00.0", "0x10de", "0x2684")
      self.add_pci_device(root, "0000:04:00.0", "not-hex", "0x0001")
      (root / "0000:05:00.0").mkdir()
      found = discover_sohu_devices(root)
      self.assertEqual([device.bdf for device in found], ["0000:03:00.0", "0000:81:00.0"])
      self.assertEqual([device.vendor_id for device in found], [0x20A1, 0x20A1])
      self.assertEqual([device.device_id for device in found], [0x0001, 0x0001])
      self.assertTrue(all(device.sysfs_path.is_dir() for device in found))

  def test_missing_root_is_an_empty_result(self):
    with tempfile.TemporaryDirectory() as tmp:
      self.assertEqual(discover_sohu_devices(Path(tmp) / "missing"), ())


class TestSohuUringCommand(unittest.TestCase):
  def test_byte_exact_linux_sqe(self):
    command = SohuUringCommand(fd=7, cmd_op=0x11223344, addr=0x0102030405060708, length=0xAABBCCDD,
                               user_data=0xFFEEDDCCBBAA9988, fixed_file=True)
    sqe = command.to_sqe()
    expected = bytearray(64)
    struct.pack_into("<B", expected, 0, 46)
    struct.pack_into("<B", expected, 1, 1)
    struct.pack_into("<i", expected, 4, 7)
    struct.pack_into("<I", expected, 8, 0x11223344)
    struct.pack_into("<Q", expected, 16, 0x0102030405060708)
    struct.pack_into("<I", expected, 24, 0xAABBCCDD)
    struct.pack_into("<Q", expected, 32, 0xFFEEDDCCBBAA9988)
    self.assertEqual(sqe, bytes(expected))
    self.assertEqual(len(sqe), 64)

  def test_direct_fd_does_not_set_fixed_file_flag(self):
    sqe = SohuUringCommand(fd=3, cmd_op=4, addr=5, length=6, fixed_file=False).to_sqe()
    self.assertEqual(sqe[1], 0)

  def test_validates_all_field_ranges(self):
    valid = {"fd": 3, "cmd_op": 4, "addr": 5, "length": 6, "user_data": 7, "fixed_file": True}
    invalid = [("fd", -(1 << 31)-1), ("fd", 1 << 31), ("fd", True), ("cmd_op", -1), ("cmd_op", 1 << 32),
               ("addr", -1), ("addr", 1 << 64), ("length", 0), ("length", 1 << 32), ("user_data", -1),
               ("user_data", 1 << 64), ("fixed_file", 1)]
    for field, value in invalid:
      with self.subTest(field=field, value=value), self.assertRaises((TypeError, ValueError)):
        SohuUringCommand(**(valid | {field: value}))


class TestLinuxSohuTransport(unittest.TestCase):
  def test_fails_closed_on_platform_device_and_node(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      with self.assertRaisesRegex(EtchedHardwareUnavailable, "Linux"):
        LinuxSohuTransport(root / "sohu0", sysfs_root=root, platform="darwin")
      with self.assertRaisesRegex(EtchedHardwareUnavailable, "20a1:0001"):
        LinuxSohuTransport(root / "sohu0", sysfs_root=root, platform="linux")
      TestSohuDiscovery.add_pci_device(root, "0000:03:00.0", "0x20a1", "0x0001")
      with self.assertRaisesRegex(EtchedHardwareUnavailable, "device node"):
        LinuxSohuTransport(root / "sohu0", sysfs_root=root, platform="linux")

  def test_selects_device_but_refuses_to_guess_vendor_abi(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      TestSohuDiscovery.add_pci_device(root, "0000:03:00.0", "0x20a1", "0x0001")
      node = root / "sohu0"
      node.touch()
      transport = LinuxSohuTransport(node, sysfs_root=root, platform="linux")
      self.assertEqual(transport.device.bdf, "0000:03:00.0")
      self.assertEqual(transport.device_node, node)
      with self.assertRaisesRegex(ValueError, "nonempty"):
        transport.submit(b"")
      with self.assertRaisesRegex(EtchedPublicSpecIncomplete, "cmd_op.*payload schema.*memory registration"):
        transport.submit(b"public-executable")


class FakeUringSystem:
  def __init__(self, result:int=0, *, complete:bool=True, valid_params:bool=True, fail_register:bool=False,
               fail_unregister:bool=False, fail_map_offset:int|None=None):
    self.result, self.complete, self.valid_params = result, complete, valid_params
    self.fail_register, self.fail_unregister, self.fail_map_offset = fail_register, fail_unregister, fail_map_offset
    self.ring:bytearray|None = None
    self.sqes:bytearray|None = None
    self.registered:tuple[int, ...]|None = None
    self.enter_calls:list[tuple[int, int, int, int]] = []
    self.unregister_count, self.closed_fd = 0, None

  def setup(self, entries:int) -> tuple[int, bytes]:
    params = bytearray(120 if self.valid_params else 12)
    if not self.valid_params: return 91, bytes(params)
    struct.pack_into("<II", params, 0, entries, entries)
    struct.pack_into("<I", params, 20, 1)  # IORING_FEAT_SINGLE_MMAP
    struct.pack_into("<IIIIIII", params, 40, 0, 4, 8, 12, 16, 20, 24)
    struct.pack_into("<IIIIIII", params, 80, 64, 68, 72, 76, 80, 96, 84)
    return 91, bytes(params)

  def map(self, fd:int, size:int, offset:int):
    self.assert_fd(fd)
    if offset == self.fail_map_offset: raise OSError(12, "fake mmap failure")
    if offset == 0:
      self.ring = bytearray(size)
      struct.pack_into("<II", self.ring, 8, 3, 4)
      struct.pack_into("<II", self.ring, 72, 3, 4)
      return self.ring
    if offset == 0x10000000:
      self.sqes = bytearray(size)
      return self.sqes
    raise AssertionError(f"unexpected mmap offset {offset:#x}")

  @staticmethod
  def assert_fd(fd:int):
    if fd != 91: raise AssertionError(f"unexpected ring fd {fd}")

  def register_files(self, fd:int, files:tuple[int, ...]):
    self.assert_fd(fd)
    if self.fail_register: raise OSError(22, "fake registration failure")
    self.registered = files

  def unregister_files(self, fd:int):
    self.assert_fd(fd)
    self.unregister_count += 1
    if self.fail_unregister: raise OSError(5, "fake unregister failure")

  def enter(self, fd:int, to_submit:int, min_complete:int, flags:int) -> int:
    self.assert_fd(fd)
    self.enter_calls.append((fd, to_submit, min_complete, flags))
    if self.complete:
      assert self.ring is not None and self.sqes is not None
      sq_tail, sq_mask = struct.unpack_from("<II", self.ring, 4)[0], struct.unpack_from("<I", self.ring, 8)[0]
      array_index = (sq_tail - 1) & sq_mask
      sqe_index = struct.unpack_from("<I", self.ring, 24 + array_index * 4)[0]
      user_data = struct.unpack_from("<Q", self.sqes, sqe_index * 64 + 32)[0]
      cq_tail, cq_mask = struct.unpack_from("<I", self.ring, 68)[0], struct.unpack_from("<I", self.ring, 72)[0]
      struct.pack_into("<QiI", self.ring, 96 + (cq_tail & cq_mask) * 16, user_data, self.result, 0xA5)
      struct.pack_into("<I", self.ring, 68, cq_tail + 1)
      struct.pack_into("<I", self.ring, 0, sq_tail)
    return to_submit

  def close_fd(self, fd:int):
    self.assert_fd(fd)
    self.closed_fd = fd


class TestLinuxIoUring(unittest.TestCase):
  def test_register_submit_complete_and_close(self):
    system = FakeUringSystem()
    with LinuxIoUring(entries=4, system=system) as ring:
      ring.register_files((37,))
      command = SohuUringCommand(fd=0, cmd_op=9, addr=0x12340000, length=128, user_data=0xCAFE)
      completion = ring.submit_sqe(command.to_sqe())
      self.assertEqual((completion.user_data, completion.result, completion.flags), (0xCAFE, 0, 0xA5))
      self.assertEqual(system.registered, (37,))
      self.assertEqual(system.enter_calls, [(91, 1, 1, 1)])
      assert system.sqes is not None
      self.assertEqual(bytes(system.sqes[:64]), command.to_sqe())
    self.assertEqual((system.unregister_count, system.closed_fd), (1, 91))

  def test_returns_negative_device_completion(self):
    with LinuxIoUring(entries=4, system=FakeUringSystem(result=-5)) as ring:
      completion = ring.submit_sqe(SohuUringCommand(fd=4, cmd_op=1, addr=2, length=3, fixed_file=False).to_sqe())
      self.assertEqual(completion.result, -5)

  def test_rejects_full_ring_bad_sqe_and_missing_completion(self):
    system = FakeUringSystem()
    with LinuxIoUring(entries=4, system=system) as ring:
      with self.assertRaisesRegex(ValueError, "64 bytes"): ring.submit_sqe(b"short")
      assert system.ring is not None
      struct.pack_into("<II", system.ring, 0, 0, 4)
      with self.assertRaises(IoUringQueueFull): ring.submit_sqe(bytes(64))

    with LinuxIoUring(entries=4, system=FakeUringSystem(complete=False)) as ring:
      with self.assertRaisesRegex(IoUringError, "completion"):
        ring.submit_sqe(SohuUringCommand(fd=4, cmd_op=1, addr=2, length=3, fixed_file=False).to_sqe())

  def test_invalid_kernel_layout_closes_ring_fd(self):
    system = FakeUringSystem(valid_params=False)
    with self.assertRaisesRegex(IoUringError, "parameter"):
      LinuxIoUring(entries=4, system=system)
    self.assertEqual(system.closed_fd, 91)

  def test_mapping_and_unregister_failures_still_close_ring_fd(self):
    mapping_failure = FakeUringSystem(fail_map_offset=0x10000000)
    with self.assertRaisesRegex(OSError, "mmap"):
      LinuxIoUring(entries=4, system=mapping_failure)
    self.assertEqual(mapping_failure.closed_fd, 91)

    unregister_failure = FakeUringSystem(fail_unregister=True)
    ring = LinuxIoUring(entries=4, system=unregister_failure)
    ring.register_files((37,))
    with self.assertRaisesRegex(OSError, "unregister"):
      ring.close()
    self.assertEqual(unregister_failure.closed_fd, 91)

  @unittest.skipUnless(sys.platform.startswith("linux"), "requires a Linux io_uring kernel")
  def test_real_linux_nop(self):
    try:
      with LinuxIoUring(entries=2) as ring:
        sqe = bytearray(64)
        struct.pack_into("<Q", sqe, 32, 0x1234)
        self.assertEqual(ring.submit_sqe(bytes(sqe)).result, 0)
    except OSError as exc:
      if exc.errno in (1, 13, 38): self.skipTest(f"io_uring unavailable in this Linux sandbox: {exc}")
      raise


class TestLinuxSohuRawSubmission(unittest.TestCase):
  def test_explicit_raw_command_uses_registered_device_and_completion(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      TestSohuDiscovery.add_pci_device(root, "0000:03:00.0", "0x20a1", "0x0001")
      node = root / "sohu0"
      node.touch()
      system, closed = FakeUringSystem(), []
      transport = LinuxSohuTransport(node, sysfs_root=root, platform="linux",
                                     ring_factory=lambda: LinuxIoUring(entries=4, system=system),
                                     device_opener=lambda path: 37, device_closer=closed.append)
      completion = transport.submit_raw(cmd_op=0x44, payload=b"abc", user_data=0x55)
      self.assertEqual((completion.user_data, completion.result), (0x55, 0))
      self.assertEqual(system.registered, (37,))
      assert system.sqes is not None
      self.assertEqual(struct.unpack_from("<I", system.sqes, 8)[0], 0x44)
      self.assertNotEqual(struct.unpack_from("<Q", system.sqes, 16)[0], 0)
      self.assertEqual(struct.unpack_from("<I", system.sqes, 24)[0], 3)
      transport.close()
      self.assertEqual(closed, [37])

  def test_negative_raw_completion_becomes_oserror(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      TestSohuDiscovery.add_pci_device(root, "0000:03:00.0", "0x20a1", "0x0001")
      node = root / "sohu0"
      node.touch()
      transport = LinuxSohuTransport(node, sysfs_root=root, platform="linux",
                                     ring_factory=lambda: LinuxIoUring(entries=4, system=FakeUringSystem(result=-5)),
                                     device_opener=lambda path: 37, device_closer=lambda fd: None)
      with self.assertRaisesRegex(OSError, os.strerror(5)):
        transport.submit_raw(cmd_op=1, payload=b"x")
      transport.close()

  def test_registration_failure_closes_ring_and_device_fds(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      TestSohuDiscovery.add_pci_device(root, "0000:03:00.0", "0x20a1", "0x0001")
      node = root / "sohu0"
      node.touch()
      system, closed, rings = FakeUringSystem(fail_register=True), [], []
      def make_ring():
        rings.append(ring:=LinuxIoUring(entries=4, system=system))
        return ring
      transport = LinuxSohuTransport(node, sysfs_root=root, platform="linux",
                                     ring_factory=make_ring,
                                     device_opener=lambda path: 37, device_closer=closed.append)
      with self.assertRaisesRegex(OSError, "registration"):
        transport.submit_raw(cmd_op=1, payload=b"x")
      self.assertEqual((system.closed_fd, closed), (91, [37]))

  def test_unconfirmed_payload_remains_pinned_until_transport_close(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      TestSohuDiscovery.add_pci_device(root, "0000:03:00.0", "0x20a1", "0x0001")
      node = root / "sohu0"
      node.touch()
      transport = LinuxSohuTransport(node, sysfs_root=root, platform="linux",
                                     ring_factory=lambda: LinuxIoUring(entries=4, system=FakeUringSystem(complete=False)),
                                     device_opener=lambda path: 37, device_closer=lambda fd: None)
      with self.assertRaises(IoUringError): transport.submit_raw(cmd_op=1, payload=b"still-in-flight")
      self.assertEqual(transport.pending_payload_count, 1)
      transport.close()
      self.assertEqual(transport.pending_payload_count, 0)


if __name__ == "__main__": unittest.main()
