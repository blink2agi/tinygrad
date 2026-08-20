"""End-to-end demonstration of the public Etched Tinygrad reference backend."""
from tinygrad import Device, Tensor


def main():
  if Device.DEFAULT != "ETCHED": raise SystemExit("run with: DEV=ETCHED python3 examples/etched_demo.py")
  left = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  right = Tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
  result = (left @ right).realize().tolist()
  driver = Device["ETCHED"].driver

  print(f"device={Device.DEFAULT}")
  print(f"result={result}")
  print(f"submissions={len(driver.submissions)}")
  print(f"completions={len(driver.completions)}")
  print(f"all_ok={all(completion.ok for completion in driver.completions)}")
  print("mode=public-reference (physical A0 ABI not publicly available)")


if __name__ == "__main__": main()
