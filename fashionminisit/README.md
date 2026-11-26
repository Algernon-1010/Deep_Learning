# Fashion-MNIST MLP 分类

Fashion-MNIST 多层感知机实验子项目。使用 PyTorch 实现从零编写的 MLP 前向传播、训练、评估与绘图流程，包含可复现的基线结果报告。

## 运行

安装 `requirements.txt` 中的依赖后，在此目录运行：

```bash
python main.py
```

查看全部参数：

```bash
python mlp_lab.py --help
```

程序以脚本所在目录作为项目根目录，将数据和新实验结果保存到本目录下。首次运行会下载 Fashion-MNIST，并校验文件完整性。数据集和训练输出不随 Git 提交。

## 基线

784→256→10 MLP，训练 10 轮，随机种子 42；测试准确率 85.04%，Macro-F1 为 0.8494。完整实验设置、分类分析和结果见 [基线实验结果](基线实验结果.md)，关键图表在 `results/`。

## 文件

- `main.py`：Fashion-MNIST 默认实验入口。
- `run_fashion.py`：兼容运行入口。
- `mlp_lab.py`：模型、训练、数据下载校验、评估与绘图实现。
- `requirements.txt`：Python 依赖版本。
