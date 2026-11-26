"""第3题 Fashion-MNIST 实验入口，可直接在 PyCharm 中运行。"""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    sys.argv[1:1] = ["--task", "fashion"]
    runpy.run_path(str(Path(__file__).resolve().with_name("mlp_lab.py")), run_name="__main__")
