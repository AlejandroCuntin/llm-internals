import os
import subprocess
import sys


TESTS = ["test_engine.py", "test_nn.py", "test_optim.py"]

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    failed = 0
    for name in TESTS:
        path = os.path.join(HERE, name)
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        result = subprocess.run([sys.executable, path])
        if result.returncode != 0:
            failed += 1

    print(f"\n{'=' * 60}")
    if failed == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {failed} test file(s) failed")
    print(f"{'=' * 60}")

    sys.exit(failed)


if __name__ == "__main__":
    main()