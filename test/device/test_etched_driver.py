import hashlib, json, struct, unittest

from tinygrad.runtime.support.etched import EtchedFormatError, PublicExecutable, PublicMatmulIR


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


if __name__ == "__main__": unittest.main()
