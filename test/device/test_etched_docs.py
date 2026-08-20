import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestEtchedDocumentation(unittest.TestCase):
  def test_public_documentation_records_interface_limits(self):
    documentation = (ROOT / "docs/etched.md").read_text()
    required = [
      "DEV=ETCHED python3 examples/etched_demo.py", "20a1:0001", "IORING_OP_URING_CMD", "cmd_op", "payload schema",
      "memory registration", "does not claim physical A0 execution", "wc -l", "aa7d43725fbf828d11fdab4356c8be90021cb6c9",
      "US20250138820A1", "US20250156164A1", "LinuxIoUring", "io_uring_setup", "submit_raw",
    ]
    for text in required:
      with self.subTest(text=text): self.assertIn(text, documentation)

  def test_notice_records_clean_room_sources(self):
    notice = (ROOT / "NOTICE.etched-public-sources.md").read_text()
    for text in ["No code from an Etched SDK", "etched-ai/io-uring", "pci.ids", "patent"]:
      with self.subTest(text=text): self.assertIn(text, notice)

  def test_readme_links_backend_and_demo_uses_etched_device(self):
    self.assertIn("[Etched public reference backend](docs/etched.md)", (ROOT / "README.md").read_text())
    demo = (ROOT / "examples/etched_demo.py").read_text()
    for text in ["Device.DEFAULT", "ETCHED", "driver.submissions", "driver.completions"]:
      with self.subTest(text=text): self.assertIn(text, demo)


if __name__ == "__main__": unittest.main()
