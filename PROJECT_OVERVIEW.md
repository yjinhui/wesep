# WeSep 项目详细解读

> 一个面向「鸡尾酒会问题」的可扩展、可泛化目标说话人提取（Target Speaker Extraction, TSE）工具包
>
> 论文：Wang et al., *WeSep: A Scalable and Flexible Toolkit Towards Generalizable Target Speaker Extraction*, Interspeech 2024.

---

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 顶层目录结构](#2-顶层目录结构)
- [3. 核心思想与任务定义](#3-核心思想与任务定义)
- [4. 系统整体架构](#4-系统整体架构)
- [5. 模型层 (wesep/models)](#5-模型层-wesepmodels)
  - [5.1 BSRNN（频段 Split Band RNN）](#51-bsrnn频段-split-band-rnn)
  - [5.2 ConvTasNet + SpEx+（时域卷积分离）](#52-convtasnet--spex时域卷积分离)
  - [5.3 DPCCN（Densely-Connected Pyramid Complex Convolution Network）](#53-dpccndensely-connected-pyramid-complex-convolution-network)
  - [5.4 TFGridNet（时频网格网络）](#54-tfgridnet时频网格网络)
  - [5.5 模型族扩展](#55-模型族扩展)
- [6. 通用基础模块 (wesep/modules)](#6-通用基础模块-wesepmodules)
  - [6.1 说话人嵌入融合（SpeakerFuseLayer）](#61-说话人嵌入融合speakerfuselayer)
  - [6.2 说话人嵌入变换（SpeakerTransform）](#62-说话人嵌入变换speakertransform)
  - [6.3 FiLM 与 ConditionalLayerNorm](#63-film-与-conditionallayernorm)
  - [6.4 归一化层（cLN / gLN / BN）](#64-归一化层cln--gln--bn)
  - [6.5 预加重 PreEmphasis](#65-预加重-preemphasis)
- [7. 数据管线 (wesep/dataset)](#7-数据管线-wesepdataset)
  - [7.1 IterableDataset / Processor 模式](#71-iterabledataset--processor-模式)
  - [7.2 数据格式 (shard / raw) 与打包](#72-数据格式-shard--raw-与打包)
  - [7.3 关键 Processor 列表](#73-关键-processor-列表)
  - [7.4 动态数据仿真（在线混合）](#74-动态数据仿真在线混合)
  - [7.5 语音活动检测 (VAD)](#75-语音活动检测-vad)
  - [7.6 数据批整理 (Collate)](#76-数据批整理-collate)
- [8. 训练与调度 (wesep/utils、wesep/bin)](#8-训练与调度-weseputilswesepbin)
  - [8.1 训练入口 train.py](#81-训练入口-trainpy)
  - [8.2 GAN 训练入口 train_gan.py](#82-gan-训练入口-train_ganpy)
  - [8.3 Executor 训练/校验循环](#83-executor-训练校验循环)
  - [8.4 损失函数 losses.py](#84-损失函数-lossespy)
  - [8.5 学习率调度器 schedulers.py](#85-学习率调度器-schedulerspy)
  - [8.6 检查点与模型平均](#86-检查点与模型平均)
- [9. 推理与评估 (wesep/bin/infer.py、wesep/bin/score.py)](#9-推理与评估-wesepbininferpywesepbinscorepy)
- [10. CLI 与预训练模型 Hub (wesep/cli)](#10-cli-与预训练模型-hub-wesepcli)
- [11. C++ 运行时 (runtime/)](#11-c-运行时-runtime)
- [12. 复现脚本与示例 (examples/、tools/)](#12-复现脚本与示例-examplestools)
- [13. 数据流图](#13-数据流图)
- [14. 安装与依赖](#14-安装与依赖)
- [15. 总结：项目亮点与适用场景](#15-总结项目亮点与适用场景)

---

## 1. 项目概览

WeSep（由 wenet-e2e 团队开发）是一个 **以目标说话人提取 (TSE) 为核心** 的开源 PyTorch 工具包。它解决的问题是 **鸡尾酒会问题**：当多个人同时讲话时，从混合语音中提取出特定说话人的声音。WeSep 与传统语音分离的最大不同在于——它只需要一段目标说话人的参考语音（enrollment），而不需要事先知道有多少个说话人。

它和 WeNet（语音识别）、WeSpeaker（说话人识别）共享同一套工具链思想：**模块化数据流水线（Processor 链）** + **可插拔模型** + **标准化 Recipe 流程** + **训练 / 推理 / 部署一体化**。

代码版本核心特点：

- **模型**：支持时域（ConvTasNet/SpEx+）和频域（BSRNN、DPCCN、TFGridNet）四类主流 TSE 架构
- **说话人建模**：集成 WeSpeaker 的预训练说话人编码器（ResNet/ECAPA-TDNN/CAMPPlus），支持多种融合方式
- **数据**：支持预混合数据集（如 Libri2Mix）和在线动态混合（VoxCeleb）两种训练模式
- **数据增强**：动态混响、动态噪声、随机 SNR、SpecAug
- **训练策略**：联合训练（joint training）/ 解耦训练 / 多任务（multi-task）/ SSA 自监督
- **损失**：SISNR、PIT、STFT、MultiResolutionSTFT、CE、GAN
- **部署**：Python CLI + C++ LibTorch 运行时，支持 JIT 导出
- **预训练**：自动下载官方 pretrained 模型

---

## 2. 顶层目录结构

```
wesep/
├── README.md                 项目主说明
├── PROJECT_OVERVIEW.md       本文档（项目详细解读）
├── setup.py                  Python 包打包 & CLI 入口注册
├── requirements.txt          完整 Python 依赖
├── .clang-format/.flake8/    代码风格配置
├── .pre-commit-config.yaml   pre-commit 钩子
│
├── wesep/                    主 Python 包
│   ├── __init__.py           暴露 load_model / load_model_local
│   ├── bin/                  可执行入口：训练 / 推理 / 评测 / 导出
│   │   ├── train.py          主训练入口
│   │   ├── train_gan.py      带 GAN 判别器的训练入口
│   │   ├── infer.py          离线推理 + SI-SNR 评估
│   │   ├── score.py          完整增强指标评测（PESQ、STOI、SISNR、DNSMOS 等）
│   │   ├── average_model.py  模型参数平均（avg_best_model）
│   │   └── export_jit.py     TorchScript 导出
│   ├── cli/                  命令行封装 + Hub
│   │   ├── extractor.py      Extractor 类（供 Python 调用）
│   │   ├── hub.py            预训练模型下载
│   │   └── utils.py          CLI 参数
│   ├── dataset/              数据管线
│   │   ├── dataset.py        IterableDataset + collate_fn
│   │   ├── processor.py      各种 processor (Wenet 风格)
│   │   ├── FRAM_RIR.py       房间冲击响应仿真
│   │   ├── vad.py            能量阈值 VAD
│   │   └── lmdb_data.py      LMDB 噪声库访问
│   ├── models/               各 TSE 模型实现
│   │   ├── __init__.py       get_model() 工厂
│   │   ├── sep_model.py      模型工厂（更新版）
│   │   ├── convtasnet.py     时域 ConvTasNet + SpEx+
│   │   ├── bsrnn.py          频域 BSRNN
│   │   ├── bsrnn_feats.py    BSRNN + 特征输入变体
│   │   ├── bsrnn_multi_optim.py  BSRNN + 多任务优化
│   │   ├── dpccn.py          DPCCN
│   │   └── tfgridnet.py      TF-GridNet
│   ├── modules/              通用子模块（供 models 复用）
│   │   ├── common/           norm.py / speaker.py / FiLM / PreEmphasis
│   │   ├── tasnet/           ConvTasNet 的 encoder/decoder/separator
│   │   ├── dpccn/            DPCCN 的 Conv / Dense / TCN 块
│   │   ├── tfgridnet/        GridNetBlock
│   │   └── metric_gan/       CMGAN 判别器
│   └── utils/                训练辅助
│       ├── executor.py       训练 / 校验循环
│       ├── executor_gan.py   GAN 版循环
│       ├── losses.py         损失注册表
│       ├── schedulers.py     学习率调度
│       ├── checkpoint.py     ckpt 加载/保存
│       ├── file_utils.py     scp / label / embedding 读取
│       ├── score.py          SI-SNR 计算
│       ├── datadir_writer.py LMDB 写入
│       ├── dnsmos.py         DNSMOS 指标
│       ├── funcs.py          fbank / CMVN / clip
│       ├── signal.py         信号处理
│       └── utils.py          日志 / 解析 / IO
│
├── runtime/                  C++ 推理运行时
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── frontend/             wav 读取 + fbank 特征管线
│   ├── separate/             SeparateEngine（封装 torch::jit::Module）
│   ├── bin/                  separate_main（命令行入口）
│   └── utils/                timer / blocking_queue / utils
│
├── tools/                    数据准备 & 评分脚本（Kaldi 风格）
│   ├── make_shard_online.py        在线 shard 打包（单说话人）
│   ├── make_shard_list_premix.py   预混合 shard 打包
│   ├── make_lmdb.py                把 MUSAN 等打成 LMDB
│   ├── extract_embed_depreciated.py
│   ├── parse_options.sh            shell 选项解析
│   ├── run.pl / split_scp.pl / score.sh / show_enh_score.sh
│   ├── print_train_val_curve.py
│   └── test_dataset.py
│
├── examples/                 复现 Recipe
│   ├── librimix/tse/v1/      Libri2Mix + LibriSpeech TSE Recipe（预混合）
│   ├── librimix/tse/v2/      Libri2Mix + LibriSpeech Recipe v2
│   └── voxceleb1/v2/         VoxCeleb1 在线混合 Recipe
│
└── resources/                README 插图
```

---

## 3. 核心思想与任务定义

**输入**：

- 混合语音波形 `x ∈ R^T`（多人讲话 + 噪声 + 混响）
- 目标说话人的一段参考语音 `e`（也称 enrollment）

**输出**：

- 仅含目标说话人的干净语音波形 `s ∈ R^T`

**Loss**：默认 SI-SDR / SI-SNR（尺度不变信噪比），也可叠加 STFT 多分辨率谱损失、GAN 判别器损失。

**两种说话人建模方式**：

1. **joint_training=True**（默认）：说话人编码器（来自 WeSpeaker）与分离网络 **端到端联合训练**，参考输入是 enrollment 语音波形
2. **joint_training=False**：使用 **预提取的说话人嵌入向量**（如 ECAPA-TDNN 提取的 192 维向量），只训练分离网络

---

## 4. 系统整体架构

```
                        ┌──────────────────────┐
   混合 wav ─────────► │  STFT / 1D-Conv     │
                        │  Encoder             │
                        └──────────┬───────────┘
                                   │   F × T 特征
                                   ▼
                        ┌──────────────────────┐
                        │  Separator Body      │ ◄── Speaker Embedding
                        │  (BSRNN/ConvTasNet/  │     (预训练 ECAPA/ResNet 或
                        │   DPCCN/TFGridNet)   │      联合训练中的 enrollment
                        │  + Speaker Fuse      │      → SpkEnc → embedding)
                        └──────────┬───────────┘
                                   │  mask / 特征
                                   ▼
                        ┌──────────────────────┐
                        │  iSTFT / 1D-Conv     │
                        │  Decoder             │
                        └──────────┬───────────┘
                                   ▼
                              输出语音
```

数据侧则采用 WeNet 风格的 Processor 链：

```
url_opener → tar_file_and_group(_single_spk) → resample → random_chunk
   → online_mix? → add_reverb? → snr_mixer → add_noise?
   → sample_enrollment → compute_fbank → apply_cmvn → spec_aug
```

---

## 5. 模型层 (wesep/models)

所有模型都遵循统一签名：

```python
class TSEModel(nn.Module):
    def forward(self, mix_wav, spk_emb):
        # mix_wav: [B, T]   spk_emb: [B, D] 或 [B, T_f, D]
        return est_wav   # [B, T]
```

`wesep/models/__init__.py`（或 `sep_model.py`）中的 `get_model(name)` 通过前缀匹配返回具体类，例如 `BSRNN`、`ConvTasNet`、`DPCCN`、`TFGridNet`，以及扩展 `BSRNN_Feats`、`BSRNN_Multi`、`CMGAN`（判别器）。

### 5.1 BSRNN（频段 Split Band RNN）

- 文件：`wesep/models/bsrnn.py`
- 频域 TSE，**官方主推模型**（README Feature List 第一项）
- 网络结构：
  - STFT（默认 win=512, stride=128）→ 把频谱切成 31 个非均匀子带（band-width 自适应，例如 0-100Hz 细，>4kHz 粗）
  - 每个子带 → BN → 1×1 Conv（→feature_dim=128）→ 堆叠 6 个 BSNet 块
  - 每个 BSNet = **band-RNN（intra-band）+ band-comm RNN（inter-band）**，均为 BiLSTM + 残差 + LayerNorm
  - 每块前接入 SpeakerFuseLayer 注入说话人嵌入（multiply/concat/additive/FiLM）
  - 子带掩码：每个子带预测 4 通道 mask（实部 / 虚部各 2 个），用 complex ratio mask 重建语音
  - iSTFT 回时域

### 5.2 ConvTasNet + SpEx+（时域卷积分离）

- 文件：`wesep/models/convtasnet.py`、`wesep/modules/tasnet/*`
- **时域 TSE**（端到端，无需 STFT）
- 三种编码器/解码器：
  - `Multi`：三个不同 kernel (16 / 80 / 160) 的 1D-Conv 并行，cLN+1×1 Conv 融合（SpEx+ 风格）
  - `Deep`：单层 Conv1D + 4 层扩张 Conv（扩张率 1/2/4/8）
  - 默认：标准 Conv1D + ReLU
- 分离块：`FuseSeparation`，由 SpeakerFuseLayer + `Separation(R, X, B, H, P)` 组成，后者是 TCN（扩张 Conv1D 块）
- 多种 spk_fuse_type：`concat`、`concatConv`、`additive`、`multiply`、`FiLM`、`None`
- `multi_fuse`：决定融合位置（每块都融还是只首块融）
- 解码器：mask → 乘回编码特征 → 转置 Conv1D（默认 `Multi` 解码）

### 5.3 DPCCN（Densely-Connected Pyramid Complex Convolution Network）

- 文件：`wesep/models/dpccn.py`、`wesep/modules/dpccn/convs.py`
- 频域，U-Net 风格 + 复数卷积
- 输入：波形 → STFT → 拆分实部/虚部 → Conv2D
- Encoder：DenseBlock + Conv2dBlock 多层下采样（16→32→64→128→384）
- 中段：Speaker 嵌入注入 + 2 层 × 10 个 TCNBlock（扩张率指数增长）
- Decoder：ConvTrans2dBlock 上采样 + DenseBlock + 金字塔池化（4/8/16/32 多尺度）
- 输出：复数掩码 → iSTFT

### 5.4 TFGridNet（时频网格网络）

- 文件：`wesep/models/tfgridnet.py`、`wesep/modules/tfgridnet/gridnet_block.py`
- 基于 ESPnet 的 TF-GridNetV2
- 频域，全局 + 子带双路径建模
- 每层：SpeakerFuse → GridNetBlock（堆叠的 DeConv1D + BiLSTM + 自注意力）
- 输入需先做 RMS 归一化（`std`），输出反归一化
- 注意：README 标注"非常慢"，实际使用需谨慎

### 5.5 模型族扩展

- `bsrnn_feats.py`：`BSRNN_Feats`，将 enrollment 直接作为 fbank 特征输入
- `bsrnn_multi_optim.py`：`BSRNN_Multi`，支持多任务、多优化器
- `modules/metric_gan/discriminator.py`：`CMGAN` 判别器（用于 GAN 训练）

---

## 6. 通用基础模块 (wesep/modules)

### 6.1 说话人嵌入融合（SpeakerFuseLayer）

`modules/common/speaker.py`：核心融合模块，被所有模型复用。

```python
SpeakerFuseLayer(embed_dim=256, feat_dim=512, fuse_type="concat")
```

支持的融合方式：

- **concat**：拼接 [x, embed] → Linear → 输出维度同 x
- **additive**：embed → Linear → 加到 x
- **multiply**：embed → Linear → 与 x 逐元素相乘
- **FiLM**：用 embed 生成 γ, β，对 x 做 `x = (1+γ)*x + β`

支持 3D `(B, C, T)` 和 4D `(B, C, T, F)` 张量。

### 6.2 说话人嵌入变换（SpeakerTransform）

3 层 1×1 Conv1d（Conv→Tanh→Conv→Tanh→Conv）的瓶颈网络，对预提取的嵌入做非线性映射，再喂给融合层。

### 6.3 FiLM 与 ConditionalLayerNorm

- `FiLM(feat_size, embed_size)`：经典 Feature-wise Linear Modulation
- `ConditionalLayerNorm`：在 LayerNorm 的 weight / bias 上做 FiLM 调制（CLN）

### 6.4 归一化层（cLN / gLN / BN）

`select_norm(norm, dim)`：返回 `ChannelWiseLayerNorm` / `GlobalChannelLayerNorm` / `BatchNorm1d`，常用于 ConvTasNet。

### 6.5 预加重 PreEmphasis

经典 `y[n] = x[n] - 0.97 * x[n-1]`，通过反向 padding + 1D Conv 实现（无学习参数），被用作说话人分支的预处理。

---

## 7. 数据管线 (wesep/dataset)

### 7.1 IterableDataset / Processor 模式

WeSep 完全沿用 WeNet 的设计（[wesep/dataset/dataset.py](wesep/dataset/dataset.py)）：

```python
class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        self.source = source
        self.f = f      # 处理函数
        ...
    def __iter__(self):
        return self.f(iter(self.source), *self.args, **self.kw)
    def apply(self, f):
        return Processor(self, f, ...)  # 链式追加
```

训练时按顺序 `dataset.apply(processor_A).apply(processor_B)...` 形成流式管道。
配合 `DistributedSampler` 支持多机多卡自动分片与 epoch 内 shuffle。

### 7.2 数据格式 (shard / raw) 与打包

两种数据来源：

- **shard**：将 wav 文件按 1000 条打包成 tar 文件（`tools/make_shard_online.py` 单说话人；`tools/make_shard_list_premix.py` 预混合），按 `utt.spk`、`utt.wav`、`utt.mix.wav` 命名
- **raw**：JSON Lines，每行 `{key, wav, spk}` 或 `{key, wav_mix, wav_spk1, wav_spk2, spk1, spk2}`

### 7.3 关键 Processor 列表

| Processor                | 功能                                                         |
| ------------------------ | ------------------------------------------------------------ |
| `url_opener`             | 本地/HTTP 打开文件流                                          |
| `tar_file_and_group`     | 解析 tar，按 prefix 聚合为一条 `{wav_mix, wav_spk1, ...}`    |
| `tar_file_and_group_single_spk` | 单说话人版（用于在线混合）                          |
| `parse_raw`              | 解析 JSON Lines                                              |
| `filter_len`             | 过滤过短/过长音频                                             |
| `shuffle`                | 局部 shuffle（buffer shuffle）                                |
| `resample`               | 统一采样率到 16 kHz                                           |
| `random_chunk`           | 随机切 3 秒（默认 `chunk_len = 16000*3`）                    |
| `mix_speakers`           | 在线动态混合：从 buffer 中随机挑干扰说话人                    |
| `snr_mixer`              | 按 SNR（默认 0 dB 或随机 -10~10 dB）合成混合                 |
| `add_noise`              | 从 LMDB 读 MUSAN 噪声叠加                                    |
| `add_reverb`             | 调用 `FRAM_RIR.single_channel()` 生成房间冲击响应并卷积      |
| `sample_spk_embedding`   | 从 `spk2embed_dict` 随机采样预提取嵌入                        |
| `sample_enrollment`      | 随机采样一段 enrollment wav（joint training 模式）            |
| `compute_fbank`          | 对 enrollment 抽 80 维 fbank                                  |
| `apply_cmvn`             | cepstral mean（subtract mean）归一化                          |
| `spec_aug`               | SpecAugment                                                  |
| `get_random_chunk`       | 辅助函数：保证所有 wav 长度一致                               |

### 7.4 动态数据仿真（在线混合）

这是 WeSep 的重要卖点。

- `mix_speakers`：维护一个 `buf`（默认 1000 条），当 buf 满时打乱顺序，循环取出每条样本作为 target，从 buf 中选另一个说话人作为干扰；支持任意 `num_speakers`
- `snr_mixer`：用目标语音的能量归一化干扰语音（再加 SNR 偏移），最后整体除以最大幅值防溢出
- `add_reverb`：调用 `dataset/FRAM_RIR.py` 中实现的快速房间冲击响应生成器（无需 RIR 库），参数 `simu_config` 控制房间大小 / RT60 / mic 距离
- `add_noise`：从 LMDB 中读 MUSAN，按概率混入

`examples/voxceleb1/v2/run_online.sh` 是典型用例：

```bash
python tools/make_shard_online.py ...   # 把 VoxCeleb1 单说话人打包
torchrun wesep/bin/train.py \
  --dataset_args.online_mix true \
  --dataset_args.noise_prob 0.25 \
  --dataset_args.reverb_prob 0.25 \
  --dataset_args.specaug_enroll_prob 0.25 \
  ...
```

### 7.5 语音活动检测 (VAD)

`dataset/vad.py`：基于短时能量的简易 VAD，按 4 秒切片、子段 100 ms 计算能量，低于 25% 分位数则判为静音，用于自动剔除 enrollment 中的静音段。
CLI 中使用更现代的 **Silero VAD**（通过 `silero-vad` 包），更鲁棒。

### 7.6 数据批整理 (Collate)

`dataset/dataset.py` 提供两个 collate 函数：

- `tse_collate_fn(batch)`：通用版，支持任意说话人数
- `tse_collate_fn_2spk(batch)`：固定 2 说话人（验证/推理用）

每个样本中的 `wav_mix` 会被复制 N 次（每个目标说话人一次），embedding 按长度做 padding / 截断。

---

## 8. 训练与调度 (wesep/utils、wesep/bin)

### 8.1 训练入口 train.py

`wesep/bin/train.py` 是主入口（用 `fire.Fire` 暴露为 CLI）：

```bash
torchrun --nproc_per_node=$N wesep/bin/train.py --config confs/xxx.yaml \
  --exp_dir exp/my_exp --gpus "[0,1]" --num_avg 10 ...
```

关键流程：

1. 解析 yaml 配置（支持命令行覆盖）
2. 初始化分布式（`torch.distributed.init_process_group("nccl")`）
3. 设置 logger + seed
4. 加载 / 解析 speaker embedding 或 spk2enroll json
5. 构造训练集与验证集
6. 构建模型（`get_model`）→ 包装为 DDP
7. 构建 optimizer（默认 Adam）+ scheduler（默认 `ExponentialDecrease`）
8. 加载 checkpoint / model_init
9. 进入 `Executor.train()` 训练循环，每个 epoch 后 `Executor.cv()` 验证
10. 保存 checkpoint（最近 N 个 + `latest_checkpoint.pt` 软链）
11. 自动绘制 loss 曲线

`find_unused_parameters=True` 用于联合训练中部分参数（如说话人分支冻结）不参与梯度计算的情况。

### 8.2 GAN 训练入口 train_gan.py

`wesep/bin/train_gan.py` 在主训练之外增加了：

- 判别器（来自 `wesep/modules/metric_gan/discriminator.py`）
- 判别器优化器 + scheduler（独立学习率）
- 损失：`loss_g = L_sisdr + λ_adv * L_adv`、`loss_d = L_d_real + L_d_fake`
- 通过 `utils/executor_gan.py` 控制交替更新

### 8.3 Executor 训练/校验循环

`utils/executor.py` 的核心 `Executor.train()`：

```python
with torch.cuda.amp.autocast(enabled=enable_amp):
    if SSA_enroll_prob > 0 and random.random() < SSA_enroll_prob:
        # SSA (Self-Supervised Augmentation): 用上一轮自己的输出作为 enrollment
        with torch.no_grad():
            est = model(features, enroll)[0]
            self_fbank = compute_fbank(est, ...)
            self_fbank = apply_cmvn(self_fbank)
        outputs = model(features, self_fbank)
    else:
        outputs = model(features, enroll)
    loss = sum(w * criterion(output, target) for ...)
scaler.scale(loss).backward()
clip_gradients(model, clip_grad)
scaler.step(optimizer); scaler.update()
scheduler.step(cur_iter)
```

`SSA_enroll_prob` 是训练 Trick：让模型把自己分离出的语音当作 enrollment 再分一次，迫使它学到不依赖高质量 enrollment 的鲁棒表示。

### 8.4 损失函数 losses.py

通过 `parse_loss(name)` 字符串查找：

| 名字                       | 来源                                                    |
| -------------------------- | ------------------------------------------------------- |
| `L1` / `L2` / `CE`         | `torch.nn`                                              |
| `PIT`                      | `torchmetrics.audio.PermutationInvariantTraining(SISNR)` |
| `STFT` / `MultiResolutionSTFT` | `auraloss.freq`                                    |
| `SISDR` / `SISNR` / `SNR`  | `auraloss.time`                                         |

`loss_posi` / `loss_weight` 数组支持对多个输出（如多任务）施加不同损失和权重。

### 8.5 学习率调度器 schedulers.py

`utils/schedulers.py`：

- `ExponentialDecrease`：warmup → 指数衰减到 `final_lr`
- 其它 `WarmupLR`、`CosineAnnealingLR` 等包装 PyTorch 原生 scheduler

### 8.6 检查点与模型平均

- `utils/checkpoint.py`：保存 `{model, optimizer, scheduler, scaler, epoch}`
- `bin/average_model.py`：对最后 N 个 epoch（或 best N 个）的参数做平均，生成 `avg_model.pt`，常用做法可显著提升泛化

---

## 9. 推理与评估 (wesep/bin/infer.py、wesep/bin/score.py)

### 推理 infer.py

- 单卡推理，`model.eval()` + `torch.no_grad()`
- 默认输出归一化（除以 max 振幅 × 0.9）
- 输出保存为 wav（`Utt{N}-{key}-T{spk}.wav`）
- 实时计算 **SI-SNR** 和 **SI-SNRi**，统计平均 + 接受率（`SI-SDRi > 1dB`）
- 支持 `save_wav=false` 关闭 wav 写入

### 评分 score.py

更全面的增强指标：

- **SI-SNR / SI-SNRi**
- **PESQ**（语音质量）
- **STOI**（可懂度）
- **DNSMOS**（基于 P.835 的 MOS 预测）
- 报告 per-utterance 与 corpus 级别

---

## 10. CLI 与预训练模型 Hub (wesep/cli)

`wesep/cli/extractor.py` 把整个流程封装为 `Extractor` 类，供 Python 一行调用：

```python
from wesep import load_model
extractor = load_model("english")          # 自动下载到 ~/.wesep/english
extractor.set_device("cuda:0")
extractor.set_vad(True)
speech = extractor.extract_speech(
    "mix.wav",         # 混合语音
    "enroll.wav",      # 目标说话人参考
)
soundfile.write("out.wav", speech[0], 16000)
```

CLI 也由 `setup.py` 注册为 `wesep` 命令：

```bash
wesep -t extraction -l english \
  --audio_file mix.wav --audio_file2 enroll.wav \
  --output_file out.wav --device cuda:0 --vad
```

`cli/hub.py::Hub`：

- `Assets = {"english": "bsrnn_ecapa_vox1.tar.gz"}`
- 模型下载到 `~/.wesep/<lang>/` 并解压为 `avg_model.pt + config.yaml`
- 当前只内置英文 BSRNN 预训练（基于 VoxCeleb1 + Libri2Mix 验证）

---

## 11. C++ 运行时 (runtime/)

### 目录布局

```
runtime/
├── frontend/        # wav.h, fbank.h, feature_pipeline.{h,cc}, fft.{h,cc}
├── separate/        # separate_engine.{h,cc}（C++ 封装 torch::jit::Module）
├── utils/           # blocking_queue.h / timer.h / utils.{h,cc}
├── bin/             # separate_main.cc（main 入口）
└── CMakeLists.txt   # 顶层
```

### 工作流程

1. `wesep/bin/export_jit.py`：用 `torch.jit.script(model, (mix, emb))` 把 PyTorch 模型冻结为 `scriptmodel.pt`
2. C++ 端 `SeparateEngine`：
   - `InitEngineThreads(1)`
   - `torch::jit::load(model_path)` 加载
   - `FeaturePipeline`（来自 wenet）做 fbank 抽取
   - `ApplyMean` 做 CMVN
   - 构造两个 `torch::Tensor`：`[2, T]` 的 wav 和 `[2, T_f, D]` 的 fbank
   - `model_->forward({torch_wav, torch_spk_emb_feat})` → 输出波形

`bin/separate_main.cc` 是一个命令行工具：接受 `wav_scp`（每行：`utt mix.wav spk1.wav spk2.wav`），输出每个 utt 两路分离后的 wav，并打印 RTF（实时因子）。

---

## 12. 复现脚本与示例 (examples/、tools/)

### examples/librimix/tse/v2/（预混合 Recipe）

- 使用 **Libri2Mix**（min/clean）+ LibriSpeech enrollment
- 数据组织：
  - `data/{train, dev, test}/{wav.scp, utt2spk, ...}`
  - 单说话人 wav.scp 用于 enrollment 采样
- 阶段化：`stage=1` 准备数据 → `stage=2` 打 shard → `stage=3` 训练 → `stage=4` 模型平均 → `stage=5` 测试

### examples/voxceleb1/v2/run_online.sh（在线混合 Recipe）

- 使用 **VoxCeleb1** 训练，**Libri2Mix** 做验证/测试
- 关键差异：
  - `data_type=shard`，但 `train` 用 `shard_online.list`（单说话人 tar），验证测试用普通 shard
  - `dataset_args.online_mix=true`、`use_random_snr=true`
  - 默认模型 `BSRNN`，配置见 [examples/voxceleb1/v2/confs/bsrnn_online.yaml](examples/voxceleb1/v2/confs/bsrnn_online.yaml)

### tools/

Kaldi 风格的辅助脚本：

| 脚本                          | 用途                                                |
| ----------------------------- | --------------------------------------------------- |
| `make_shard_online.py`        | 把单说话人 wav.scp 打 tar + list                   |
| `make_shard_list_premix.py`   | 把预混合 wav 打 tar + list                          |
| `make_lmdb.py`                | 把 MUSAN 噪声音频打成 LMDB（噪声检索更快）          |
| `extract_embed_depreciated.py`| 旧版：用 Kaldi 提取 x-vector（已被 WeSpeaker 替代） |
| `parse_options.sh`            | shell 选项解析                                       |
| `run.pl` / `split_scp.pl`     | 网格作业调度脚本                                     |
| `score.sh` / `show_enh_score.sh` | 评测 PESQ/STOI/SI-SNR 包装脚本                  |
| `print_train_val_curve.py`    | 可视化 loss                                         |
| `test_dataset.py`             | 单元测试：dataset pipeline                          |

---

## 13. 数据流图

```
                      ┌──────────────────────────────────┐
                      │  wav.scp / utt2spk / spk2utt      │
                      │  shard_online.list / shard.list   │
                      └────────────────┬─────────────────┘
                                       │
                       (DistributedSampler: rank/world shuffle)
                                       │
                                       ▼
                            ┌────────────────────┐
                            │ url_opener         │
                            │ tar_file_and_group │ ← 数据类型决定
                            └────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
        resample (→16k)        filter_len            shuffle (buffer)
              │                      │                      │
              └──────── random_chunk ┘                      │
                                     │                      │
                       (online_mix?) mix_speakers            │
                                     │                      │
                          add_reverb / snr_mixer             │
                                     │                      │
                            add_noise (lmdb)                │
                                     │                      │
         ┌───────────────────────────┴──────────────────────┘
         │
         │  (joint_training?)
         ├─ sample_enrollment (随机选 enrollment wav)
         │      │
         │      ▼
         │   compute_fbank → apply_cmvn → (specaug?)
         │
         └─► sample_spk_embedding (预提取向量)
                          │
                          ▼
                  ┌──────────────┐
                  │ collate_fn   │ (拼接 / pad)
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │   Dataloader │
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │  model(mix,  │
                  │  emb/spk)    │
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │  loss (SISDR │
                  │  / STFT / ..)│
                  └──────┬───────┘
                         ▼
                  back-prop → optimizer.step → scheduler.step
```

---

## 14. 安装与依赖

### 环境要求

- Python 3.9
- PyTorch ≥ 1.12.0（README 推荐 1.12.1）
- torchaudio ≥ 0.12.0
- CUDA 11.3

### 安装

```bash
conda create -n wesep python=3.9
conda activate wesep
conda install pytorch=1.12.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
pre-commit install
pip install -e .   # 注册 wesep CLI
```

### 核心依赖（requirements.txt）

- 训练：`tqdm`、`kaldiio`、`lmdb`、`numpy`、`scipy`、`soundfile`、`librosa`
- 评测：`fast_bss_eval`、`pesq`、`pystoi`、`mir_eval`、`torchmetrics`、`auraloss`
- 工具：`fire`（CLI）、`matplotlib`（loss 曲线）、`tableprint`（表格日志）、`PyYAML`
- VAD：`silero-vad==5.1.2`
- 代码规范：`flake8` 系、`pre-commit`
- C++ 端：`libtorch`、`gflags`、`glog`

---

## 15. 总结：项目亮点与适用场景

### 亮点

1. **模型丰富**：覆盖时域（ConvTasNet/SpEx+）与频域（BSRNN/DPCCN/TFGridNet）四大 TSE 架构，开箱即用
2. **灵活的目标说话人建模**：
   - 联合训练（spk encoder + separator 一起优化）
   - 解耦训练（用预提取的 ECAPA/ResNet embedding）
   - 多种融合方式（concat / additive / multiply / FiLM）
3. **可扩展的数据管线**：WeNet 风格 Processor 链，支持在线动态混合 + 噪声 + 混响 + SpecAug，几乎无限的数据多样性
4. **完整的工程链**：从 yaml Recipe → 训练 → 模型平均 → 推理 → 指标评测 → TorchScript 导出 → C++ 部署
5. **官方预训练模型**：`wesep` 命令一键加载
6. **SSA 自监督训练 Trick**：提升对低质量 enrollment 的鲁棒性

### 适用场景

- 视频会议 / 直播 / 通话中提取特定发言人
- 智能音箱 / 语音助手（鸡尾酒会问题）
- 听写 / 会议记录系统的前端预处理
- 多人对话 ASR 的前端增强
- 学术研究：模型对比、ablation、新模型接入

### 不擅长 / 暂未覆盖

- 多通道 / 麦克风阵列（仅单通道）
- 流式 / 实时推理（C++ 端为离线）
- 无 enrollment 的盲源分离（README 标注为 Future work）
- 训练数据中的说话人数 > 2 时需要更复杂的扰动 / 仿真

### 引用

```bibtex
@inproceedings{wang24fa_interspeech,
  title     = {WeSep: A Scalable and Flexible Toolkit Towards Generalizable Target Speaker Extraction},
  author    = {Shuai Wang and Ke Zhang and Shaoxiong Lin and Junjie Li and Xuefei Wang and Meng Ge and Jianwei Yu and Yanmin Qian and Haizhou Li},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4273--4277},
  doi       = {10.21437/Interspeech.2024-1840},
}
```