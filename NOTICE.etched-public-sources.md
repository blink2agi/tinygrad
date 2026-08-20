# Etched public backend: public-source notice

This maintained-fork addition was implemented from public, independently inspectable facts. No code from an Etched SDK, firmware image, kernel driver, confidential document, or the historical `etched` PyPI package was copied or linked.

## Public inputs

- The bounty statement: <https://x.com/SemiAnalysis_/status/2090287730605355069>
- Etched's public `etched-ai/io-uring` `SohuSendCmd` commit: <https://github.com/etched-ai/io-uring/commit/aa7d43725fbf828d11fdab4356c8be90021cb6c9>
- Linux `io_uring_sqe` definitions already generated in Tinygrad from the Linux UAPI.
- The public pci.ids registry identity for Etched Sohu: vendor `20a1`, device `0001`.
- Public patent applications US20250138820A1 and US20250156164A1 for semantic IR field names.
- Etched's public progress page for the existence and A0 status of Sohu silicon.
- Tinygrad's MIT-licensed device, renderer, allocator, and Python-UOp reference interfaces.

Public facts were used only to write a new Python implementation. The patent-field JSON record is labeled as a semantic reference format and is not asserted to be Etched's binary executable ABI.
