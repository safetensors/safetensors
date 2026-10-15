<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/safetensors/assets/raw/main/banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/safetensors/assets/raw/main/banner-light.svg">
    <img alt="Hugging Face Safetensors Library" src="https://huggingface.co/datasets/safetensors/assets/raw/main/banner-light.svg" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

Python
[![Pypi](https://img.shields.io/pypi/v/safetensors.svg)](https://pypi.org/pypi/safetensors/)
[![Documentation](https://img.shields.io/website/http/huggingface.co/docs/safetensors/index.svg?label=docs)](https://huggingface.co/docs/safetensors/index)
[![Downloads](https://static.pepy.tech/badge/safetensors/month)](https://pepy.tech/project/safetensors)

Rust
[![Crates.io](https://img.shields.io/crates/v/safetensors.svg)](https://crates.io/crates/safetensors)
[![Documentation](https://docs.rs/safetensors/badge.svg)](https://docs.rs/safetensors/)
[![Dependency status](https://deps.rs/repo/github/huggingface/safetensors/status.svg?path=safetensors)](https://deps.rs/repo/github/huggingface/safetensors?path=safetensors)

# safetensors

## Safetensors 简介

本代码库实现了一种全新、简单且**安全**（相较于存在任意代码执行隐患的 pickle）存储张量（Tensor）的格式，同时依然保持极高速度（零拷贝，zero-copy）。

### 安装指南
#### Pip 安装

您可以通过 pip 包管理器直接安装 safetensors：

```bash
pip install safetensors
```

#### 从源码安装

从源码编译构建需要配置 Rust 环境：

```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# 确保已更新至最新稳定版 (stable channel)
rustup update
git clone https://github.com/huggingface/safetensors
cd safetensors/bindings/python
pip install setuptools_rust
pip install -e .
```

### 快速上手

```python
import torch
from safetensors import safe_open
from safetensors.torch import save_file

tensors = {
   "weight1": torch.zeros((1024, 1024)),
   "weight2": torch.zeros((1024, 1024))
}
save_file(tensors, "model.safetensors")

tensors = {}
with safe_open("model.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)
```

[Python 官方参考文档](https://huggingface.co/docs/safetensors/index)


### 格式规范 (Format)

- 8 字节：`N`，一个小端序（little-endian）无符号 64 位整数，记录文件头部的大小；
- N 字节：JSON UTF-8 字符串，表示元数据头部；
  - 头部数据必须以 `{` 字符（0x7B）开始；
  - 头部末尾允许使用空格字符（0x20）进行尾随填充（trailing padded）；
  - 头部是一个字典结构，形如 `{"TENSOR_NAME": {"dtype": "F16", "shape": [1, 16, 256], "data_offsets": [BEGIN, END]}, "NEXT_TENSOR_NAME": {...}, ...}`；
    - `data_offsets` 指向张量数据相对于字节缓冲区开头的偏移量（即并非在文件中的绝对物理位置），其中 `BEGIN` 为起始偏移，`END` 为结束偏移（开区间，因此张量总字节大小 = `END - BEGIN`）；
  - 允许使用特殊键 `__metadata__` 存储任意的字符串键值对映射（string-to-string map）。不允许存储任意复杂 JSON 对象，所有值必须为字符串；
- 文件剩余部分：二进制数据字节缓冲区（byte-buffer）。

设计说明：
 - 禁止出现重复的键（Duplicate keys）。部分解析器可能未对此做严格校验。
 - 本库所支持的 JSON 子集通常隐式由 `serde_json` 决定。任何模棱两可的表示（例如在 UTF-8 字符串中以古怪方式表示整数、换行符和转义符）后续可能会进行规范修正，但这完全出于安全性考虑。
 - 不对张量内部数值进行有效性校验，文件中可能包含 NaN 或 +/-Inf。
 - 允许空张量（维度中包含 0 的张量）。它们不会在数据缓冲区中占用任何存储空间，但在头部中保留其尺寸信息。尽管它们不包含实际数值，但从主流张量库（PyTorch、TensorFlow、NumPy 等）的角度看它们是合法的张量，因此予以支持。
 - 允许 0 阶张量（形状为 `[]` 的张量），它们本质上是标量。
 - 字节缓冲区必须被完全连续索引，不允许出现未被引用的间隙空洞。这一设计能够彻底杜绝多义（polyglot）恶意文件的构造。
 - 字节序：小端序（Little-endian）。
 - 内存布局顺序：'C' 或行优先（row-major）。
 - 跨步（Stride）：不支持带跨步的张量，所有张量在序列化前必须是连续紧凑的（packed）。在序列化格式中存储跨步张量并无实质价值。
 - 小于 1 字节的数据类型：新出现的一些小于 1 字节的数据类型会带来复杂的对齐与寻址问题。目前若某项操作触发非对齐读取，本库将直接报错；后续可能会为这些非标准算子设计更精细的专用 API。


### 为什么需要又一种新格式？(Yet another format ?)

创建本项目的核心动机在于彻底替代 `PyTorch` 默认使用的 `pickle`。
业界和通用领域虽然存在其他格式，但各有痛点。

以下是主流替代格式的对比分析（带有个人视角与客观权衡）：

| 格式 (Format)            | 安全 (Safe) | 零拷贝 (Zero-copy) | 延迟加载 (Lazy loading) | 无文件大小限制 | 布局控制 (Layout control) | 灵活性 (Flexibility) | Bfloat16/FP8 原生支持 |
| ----------------------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| pickle (PyTorch)        | ✗ | ✗ | ✗ | 🗸 | ✗ | 🗸 | 🗸 |
| H5 (Tensorflow)         | 🗸 | ✗ | 🗸 | 🗸 | ~ | ~ | ✗ |
| SavedModel (Tensorflow) | 🗸 | ✗ | ✗ | 🗸 | 🗸 | ✗ | 🗸 |
| MsgPack (flax)          | 🗸 | 🗸 | ✗ | 🗸 | ✗ | ✗ | 🗸 |
| Protobuf (ONNX)         | 🗸 | ✗ | ✗ | ✗ | ✗ | ✗ | 🗸 |
| Cap'n'Proto             | 🗸 | 🗸 | ~ | 🗸 | 🗸 | ~ | ✗ |
| Arrow                   | ? | ? | ? | ? | ? | ? | ✗ |
| Numpy (npy,npz)         | 🗸 | ? | ? | ✗ | 🗸 | ✗ | ✗ |
| pdparams (Paddle)       | ✗ | ✗ | ✗ | 🗸 | ✗ | 🗸 | 🗸 |
| **SafeTensors**         | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | ✗ | 🗸 |

- **安全 (Safe)**：随意下载的一个外部文件是否能够保证绝不执行任意恶意代码？
- **零拷贝 (Zero-copy)**：读取文件所需的内存是否严格不超过原始文件本身的大小？
- **延迟加载 (Lazy loading)**：能否在无需加载全部内容的情况下检查文件元数据？能否只加载其中的特定张量，而无需扫描整个文件（这在分布式场景下至关重要）？
- **布局控制 (Layout control)**：仅仅支持延迟加载并不充分。若张量的数据分散在文件中各处，即便可以延迟访问，也可能需要遍历大部分文件才能读取可用张量（引发大量的磁盘到内存拷贝）。控制底层布局以确保单张量的快速定位访问极为关键。
- **无文件大小限制 (No file size limit)**：是否存在硬性文件体积上限？
- **灵活性 (Flexibility)**：能否保存自定义代码并在后续无需额外代码即可直接调用？（`~` 表示可以存储纯张量之外的更多信息，但不能执行自定义代码）
- **Bfloat16/FP8**：格式是否原生支持 bfloat16 与 fp8（无需怪异的规避方案）？这在现代机器学习领域正变得前所未有地重要。


### 主流替代方案的局限性

- **Pickle**：不安全，加载时可能执行任意恶意代码。
- **H5**：目前在 TF/Keras 中已被官方劝退。在其他方面看似不错，但历史上存在经典的释放后使用（Use-After-Free）漏洞（详见 [CVE 列表](https://www.cvedetails.com/vulnerability-list/vendor_id-15991/product_id-35054/Hdfgroup-Hdf5.html)）。在安全性维度与 pickle 不可同日而语。此外 HDF5 有超过 21 万行代码，而本库目前仅约 400 行代码。
- **SavedModel**：深度绑定 TensorFlow（包含 TF 计算图信息）。
- **MsgPack**：缺乏布局控制，无法高效支持延迟加载（这在分布式环境中按需加载指定切片时十分重要）。
- **Protobuf**：存在 2GB 的硬性文件体积上限。
- **Cap'n'proto**：缺乏对 Float16 的原生支持（参见[官方类型说明](https://capnproto.org/language.html#built-in-types)），必须在字节缓冲区上进行手动封装。此外缓冲区大小存在限制（参见[讨论](https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize)），布局控制并不容易。
- **Numpy (npz)**：缺乏 `bfloat16` 支持。易受 Zip 炸弹（Zip Bomb 拒绝服务攻击）威胁，且不支持零拷贝。
- **Arrow**：缺乏原生 `bfloat16` 支持。

### 技术细节说明 (Notes)

- **零拷贝 (Zero-copy)**：在机器学习中，没有任何格式能在所有场景下实现绝对意义的零拷贝，因为数据必须从磁盘转移到内存/显存（这本身需要时间）。在 CPU 上，若文件已被操作系统缓存在页面缓存（Page Cache）中，则可以做到真正的零拷贝；而在 GPU 上由于不存在此类磁盘缓存，拷贝是必不可少的，但 SafeTensors 允许完全绕过在 CPU 上分配所有张量的中间步骤。
  SafeTensors 对于头部并非零拷贝。采用 JSON 是一个权衡选择，因为头部反序列化所需的时间远小于加载实际张量数据的时间（占用的磁盘空间也远小于张量数据），且具备极佳的可读性。
- **字节序 (Endianness)**：采用小端序（Little-endian）。后续如果确有必要可以增加扩展，但目前看来并无实际需求。
- **内存排序 (Order)**：采用 'C' 或行优先（row-major）顺序，这已成为业界的通用标准。
- **跨步 (Stride)**：不支持跨步，所有张量在序列化前必须打平并紧凑存储。
- **小于 1 字节的数据类型 (Sub 1 bytes dtypes)**：当数据类型小于 1 字节时会带来对齐与寻址挑战。目前当某项操作触发非对齐读取时，本库会直接报错。后续可能会为这些非标准操作提供专用接口。

### 核心优势 (Benefits)

设计一种全新格式让我们能够引入更多安全与性能红利：

- **防止拒绝服务（DOS）攻击**：我们针对格式进行了精心设计，几乎杜绝了利用恶意伪造文件发起 DOS 攻击的可能。目前头部大小硬性限制在 100MB 以内，防止解析超大恶意 JSON。此外在读取文件时，严格保证文件内的各张量地址互不重叠，这意味着在加载文件时，消耗的内存绝对不会超过文件本身的实际大小。
- **极致加载速度**：在主流 ML 格式中 PyTorch 原生加载速度本已较快，但在 CPU 上往往会产生额外拷贝。在本库中，我们通过利用 `torch.UntypedStorage.from_file` 彻底省去了该拷贝。目前在 CPU 上，本库的加载速度相比 pickle 快数倍；在 GPU 上的加载速度与 PyTorch 持平或更快。通过 mmap 内存映射先在 CPU 打开再批量传输到 GPU，也能带来显著的速度提升。
- **优雅的延迟加载**：在分布式（多机或多卡）环境中，各卡只需加载模型的部分张量切片。对于 [BLOOM](https://huggingface.co/bigscience/bloom) 模型，使用 SafeTensors 格式将 8 卡 GPU 上的模型加载时间从常规 PyTorch 权重的 **10 分钟大幅缩短至 45 秒**！这极大地加速了模型开发与调试的反馈闭环，在切换并行策略（例如流水线并行 vs 张量并行）时，甚至无需为不同策略维护多份独立的模型权重。

## 开源许可证 (License)

基于 [Apache-2.0](LICENSE) 许可证开源授权。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年10月15日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
