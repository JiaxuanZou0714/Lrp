# LLaMA 预训练管线（Muon / RMNP / Shampoo / SOAP / NewOpt / NORA / Mano）

这是一个基于 PyTorch 的 LLaMA 预训练项目，支持多种优化器和多种模型规模，适配多卡分布式训练。

当前文档以仓库现有代码为准，核心入口如下：

- 主训练入口：`torchrun_main.py`
- 通用训练脚本：`scripts/train_universal.sh`
- 预设训练脚本：`scripts/train_<optimizer>_<model>.sh`
- 数据准备脚本：`prepare_data.py`

## 1. 支持范围

### 模型规模

- `60m`
- `135m`
- `350m`
- `1b`

对应配置文件在 `configs/`：

- `configs/llama_60m.json`
- `configs/llama_135m.json`
- `configs/llama_350m.json`
- `configs/llama_1b.json`

### 优化器

- `muon`
- `RMNP`
- `shampoo`
- `soap`
- `new_optimizer`
- `NORA`
- `mano`

## 2. 环境准备

### 基础依赖

- Linux + NVIDIA GPU
- Python 3.9+
- Conda（推荐）

### 安装步骤

```bash
cd LLaMA

conda create -n llama_training python=3.9 -y
conda activate llama_training

pip install -r requirements_pip.txt
```

也可以按需使用 `requirements.txt`（conda 导出风格）进行环境对齐。

## 3. 凭据配置

训练前请配置以下环境变量：

- `HF_TOKEN`（数据访问）
- `WANDB_API_KEY`（实验追踪）
- `WANDB_PROJECT`（可选，默认 `llama-pretraining`）

推荐方式：

```bash
cp .env.template .env
source .env
```

或手动导出：

```bash
export HF_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_key"
export WANDB_PROJECT="llama-pretraining"
```

## 4. 数据准备（强烈建议先做）

项目默认使用本地数据目录（`c4_local`）。建议先下载并固化本地快照：

```bash
python prepare_data.py \
  --train_examples 5200000 \
  --val_examples 40000 \
  --save_path ./c4_local
```

训练时通过 `--local_data_dir ./c4_local` 显式指定，便于复现与迁移。

如果训练速度瓶颈在实时 tokenize，建议先做一次离线预分词：

```bash
python prepare_tokenized_data.py \
  --input_data_dir ./c4_local \
  --output_data_dir ./c4_tokenized_t5_base \
  --tokenizer_name_or_path t5-base \
  --max_length 256 \
  --num_proc 16
```

训练时可直接使用 `--tokenized_data_dir ./c4_tokenized_t5_base`，从而跳过在线分词。

## 5. 快速开始

先给脚本执行权限：

```bash
chmod +x scripts/*.sh
```

### 常用预设脚本

60M：

- `scripts/train_muon_60m.sh`
- `scripts/train_RMNP_60m.sh`
- `scripts/train_shampoo_60m.sh`
- `scripts/train_soap_60m.sh`
- `scripts/train_newopt_60m.sh`
- `scripts/train_nora_60m.sh`
- `scripts/train_mano_60m.sh`

135M：

- `scripts/train_muon_135m.sh`
- `scripts/train_RMNP_135m.sh`
- `scripts/train_shampoo_135m.sh`
- `scripts/train_soap_135m.sh`

350M：

- `scripts/train_muon_350m.sh`
- `scripts/train_RMNP_350m.sh`
- `scripts/train_shampoo_350m.sh`
- `scripts/train_soap_350m.sh`

1B：

- `scripts/train_muon_1b.sh`
- `scripts/train_RMNP_1b.sh`
- `scripts/train_shampoo_1b.sh`
- `scripts/train_soap_1b.sh`

示例：

```bash
./scripts/train_muon_60m.sh
./scripts/train_RMNP_135m.sh
```

## 6. 通用脚本（推荐做自定义实验）

你可以直接使用：

```bash
./scripts/train_universal.sh \
  --model_size 135m \
  --optimizer RMNP \
  --num_gpus 4 \
  --lr_matrix 0.005 \
  --lr_adam 0.001 \
  --num_steps 20000 \
  --batch_size 64 \
  --total_batch_size 512 \
  --tokenized_data_dir ./c4_tokenized_t5_base \
  --tokenizer_name_or_path t5-base \
  --tokenizer_local_files_only
```

说明：

- `train_universal.sh` 会自动计算 `gradient_accumulation`。
- 该脚本会自动写出 `training_command.log` 和 `training_params.json`。
- 当前脚本参数校验中，`--lr_matrix` 和 `--lr_adam` 需要同时提供。

## 7. Reproducibility（复现指南）

本节整合了代码行为与项目记忆中的实战经验。

### 7.1 复现前固定项

1. 固定代码版本

```bash
git rev-parse HEAD
```

2. 固定运行环境

```bash
pip freeze > env_freeze.txt
python --version > python_version.txt
nvidia-smi > nvidia_driver.txt
```

3. 固定 Python 解释器（来自项目记忆，建议始终执行）

```bash
export PYTHON_BIN="$(which python)"
```

如果终端在 `base` 环境，显式指定解释器可以避免误用系统依赖。

4. 固定数据快照

- 使用 `prepare_data.py` 生成本地数据。
- 训练时固定 `--local_data_dir`。

### 7.2 推荐的可复现实验命令

若希望精确控制 `--seed`、`--save_dir` 等参数，建议直接调用主入口：

```bash
${PYTHON_BIN:-python} -m torch.distributed.run --standalone --nproc-per-node 4 \
  torchrun_main.py \
  --model_config configs/llama_60m.json \
  --optimizer RMNP \
  --lr 0.001 \
  --lr_matrix 0.005 \
  --lr_adam 0.005 \
  --batch_size 64 \
  --total_batch_size 512 \
  --gradient_accumulation 2 \
  --num_training_steps 10000 \
  --warmup_steps 1000 \
  --weight_decay 0.1 \
  --eval_every 1000 \
  --save_every 5000 \
  --dtype bfloat16 \
  --max_length 256 \
  --scheduler cosine \
  --grad_clipping 1.0 \
  --seed 0 \
  --local_data_dir ./c4_local \
  --save_dir ./checkpoints/repro_rmnp_60m_seed0 \
  --wandb_name repro_rmnp_60m_seed0
```

### 7.3 续训复现

```bash
${PYTHON_BIN:-python} -m torch.distributed.run --standalone --nproc-per-node 4 \
  torchrun_main.py \
  ...同一组参数... \
  --continue_from ./checkpoints/repro_rmnp_60m_seed0/model_5000
```

续训时请保持关键超参数一致（优化器、学习率、batch、调度器、数据路径）。

### 7.4 最小复现归档清单

- commit hash
- `env_freeze.txt`
- `python_version.txt`
- `nvidia_driver.txt`
- `training_command.log`
- `training_params.json`
- `model_<step>/optimizer.pt`
- `model_<step>/training_state.json`

### 7.5 结果一致性预期

- 在相同 commit、seed、数据和环境下，指标应高度接近。
- 多卡 + bf16 场景通常不保证 bitwise 完全一致。
- 对比建议至少跟踪：`final_eval_loss`、`final_perplexity`、`tokens_seen`。

## 8. 输出目录说明

训练输出默认保存在 `checkpoints/`：

```text
checkpoints/
└── llama_60m-2026-04-16-10-00-00-RMNP-matrix0.005-adam0.005/
    ├── training_command.log
    ├── training_params.json
    ├── model_5000/
    │   ├── pytorch_model.bin
    │   ├── optimizer.pt
    │   └── training_state.json
    └── model_10000/
```

## 9. 故障排查

1. CUDA OOM：降低 `--batch_size` 或 `--total_batch_size`，必要时开启 `--activation_checkpointing`。
2. 缺少 Muon 依赖：`pip install muon-optimizer`。
3. WandB 报错：检查 `WANDB_API_KEY`。
4. HuggingFace 访问异常：检查 `HF_TOKEN`。
5. CXXABI 报错（来自项目记忆中的高频问题）：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
```

6. 训练脚本走错 Python：优先设置 `PYTHON_BIN` 指向当前 conda 环境解释器。

## 10. 快速自检

```bash
./scripts/test_pipeline.sh
```

这个脚本会检查训练入口、脚本可执行性、配置文件和基础语法。