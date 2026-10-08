# mini seq2seq
使用 PyTorch 实现的简易 Seq2Seq 注意力模型，用于神经机器翻译。

本项目主要具有以下特点：

- 模块化结构，可复用于其他项目
- 代码精简，便于阅读
- 充分利用批处理和 GPU。

数据集使用 Hugging Face [`datasets`](https://github.com/huggingface/datasets) 加载 Multi30k 德译英数据；分词使用 [spaCy](https://spacy.io/)。

## 模型说明

* 编码器：双向 GRU
* 解码器：带注意力机制的 GRU
* 注意力机制论文：[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* 相关博客：[深度学习和自然语言处理中的注意力和记忆](https://dennybritz.com/posts/wildml/attention-and-memory-in-deep-learning-and-nlp/)

![attention-and-memory-in-deep-learning-and-nlp](https://dennybritz.com/wp-content/uploads/2015/12/Screen-Shot-2015-12-30-at-1.16.08-PM.png)

## 安装

需要 Python 3.9 或更高版本。运行以下命令安装项目及其依赖：

```
python -m pip install -e .
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## 训练

```
python train.py -epochs 30 -batch_size 32 -lr 3e-4
```

程序会自动检测运行设备（CUDA → MPS → CPU）。在 CPU 上快速试运行时，可适当调小 `-hidden_size` 和 `-embed_size` 参数。

## 使用 Modal 进行 GPU 训练

项目提供三个 Modal 入口：CPU 数据准备入口下载 Multi30k、缓存并编码数据集，同时将德语和英语 spaCy 模型及词表保存到 Modal Volume；GPU 训练入口读取同一 Volume 中的数据，复用项目训练循环训练模型，并保存最佳模型检查点和训练摘要；CPU 推理入口从测试集抽样，输出模型预测和参考译文。

运行前，在 Modal 中创建名为 `huggingface-secret` 的 Secret，并添加 `HF_TOKEN` 键。也可以先在本地设置 `HF_TOKEN`，再运行 `modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"`。预处理节点通过该 Secret 获取 Hugging Face 访问令牌。

在项目根目录、已配置 Modal 登录信息的终端中，使用你选择的 Python 环境安装项目依赖和命令：

```bash
python -m pip install -e .
```

安装后，`pyproject.toml` 的 `[project.scripts]` 会提供以下终端命令。它们会调用本地 `modal` 命令行工具，在 Modal 云端运行相应任务：

| 命令 | 功能 | 默认参数 |
| --- | --- | --- |
| `prepare-modal-data` | CPU 节点下载、处理数据并保存到 Volume | `bentrevett/multi30k`，数据产物名为 `multi30k` |
| `train-modal-seq2seq` | GPU 节点训练并保存模型检查点 | 训练 30 轮，每批 32 条 |
| `infer-modal-seq2seq` | CPU 节点从测试集抽样并输出预测和参考译文 | 检查点 `run-20261008T101910930522Z.pt`，抽取 10 条，随机种子为 42 |

先设置要使用的 Modal Volume 名称，然后依次运行数据准备和训练命令：

```bash
export SEQ2SEQ_MODAL_VOLUME=mini-seq2seq-data
prepare-modal-data
```

数据准备完成后，启动 GPU 训练：

```bash
train-modal-seq2seq
```

GPU 默认使用 `A10G`，可通过 `SEQ2SEQ_MODAL_GPU` 更改。训练检查点和 JSON 摘要分别写入 Volume 的 `models/<数据产物名>/` 目录。每次运行都会使用独立的检查点文件名；若要准备另一份数据，请使用不同的 `--artifact-name`。

本次训练日志显示，训练在第 14 轮提前停止，测试集损失为 `3.38`，最佳检查点为 `run-20261008T101910930522Z.pt`。运行下面的 CPU Modal 脚本，可从测试集中随机抽取 10 条德语样本进行推理，并同时输出模型预测和参考译文：

```bash
infer-modal-seq2seq
```

推理命令默认读取 `mini-seq2seq-data` Volume 中的 `models/multi30k/run-20261008T101910930522Z.pt`，并从 `artifacts/multi30k/dataset` 的测试集划分中抽样。`[project.scripts]` 命令使用默认参数；若要更换检查点或随机种子，可直接运行 `modal run infer_modal.py --checkpoint-name <文件名.pt> --seed <整数>`。

CPU 快速检查结果（500 个批次，隐藏层大小为 128，嵌入维度为 64）：

| 步数 | 训练损失 | 困惑度 |
|------|-----------:|-----------:|
| 初始 |       9.19 |      9803 |
|   50 |       6.98 |      1071 |
|  100 |       5.48 |       239 |
|  250 |       5.15 |       173 |
|  500 |       4.84 |       127 |

最终验证集损失：**4.93**（随机初始化时的基准值为 `log(|V|) ≈ 9.19`）。

## 参考资料

本项目参考了以下实现：

* [PyTorch Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
* [@spro/practical-pytorch](https://github.com/spro/practical-pytorch)
* [@AuCson/PyTorch-Batch-Attention-Seq2seq](https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq)
