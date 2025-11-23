# MatterGen Demo Toolkit

`crygen_demo/` 精简了我们在 `test/` 下构建的自动化脚本，方便直接开箱体验 MatterGen 的结构生成与微调流程。目录只保留 README 中提到的核心脚本，便于发布到 GitHub 或分发给使用者。

```
./crygen_demo
├── README.md              # 本使用说明
├── common/                # 常量、CLI、远程提交、打包共享工具
├── generation/
│   ├── local.py           # 本地生成入口
│   ├── remote.py          # 生成任务远程提交（Bohrium）
│   └── pull_results.py    # 生成任务远程结果拉取工具
├── finetune/
│   ├── local.py           # 本地微调入口
│   ├── remote.py          # 微调任务远程提交（Bohrium）
│   ├── evaluate_finetune.py
│   ├── pull_results.py
│   ├── run_finetune.py    # 兼容入口（仍调用 finetune.local）
│   └── submit_finetune.py
├── run_generation.py      # 兼容入口（调用 generation.local / fallback）
└── submit_gen.py          # 兼容入口（调用 generation.remote）
```

## 1. 环境准备

- Python 3.10+，建议直接使用官方 `mattergen` 仓库提供的虚拟环境或 conda 环境（需包含 PyTorch Lightning、Hydra、dpdispatcher 等依赖）。
- 默认路径：
    - 基线模型：`../mattergen/checkpoints/mattergen_base`（目录，可替换）
    - 本地生成输出：`crygen_demo/results`
    - 微调数据：`crygen_demo/data`（示例数据，可自定义）
    - 远程模型默认上传为 `./model` 目录
- 远程任务需要准备 Bohrium 账号、密码、program_id 以及镜像名称（默认镜像见下文）。

> **Tip**：脚本只依赖 `common/` 模块中的常量和工具，不依赖 `test/` 中的其他文件，可单独复制到任何目录使用。

## 2. 本地结构生成

### 2.1 基本命令

```bash
python run_generation.py \
    --results-dir crygen_demo/results/demo_run \
    --model-dir ../mattergen/checkpoints/mattergen_base
```

默认使用 `mattergen-generate` 执行，命令行参数：

- `--results-dir`：生成结果输出目录，默认 `crygen_demo/results`
- `--model-dir`：MatterGen 模型目录（必须包含 `config.yaml`、`checkpoints/` 等）
- `--batch-size`、`--num-batches`：每批样本数与批次数；默认 4×2
- `--num-gen`：目标结构总数（新功能）

### 2.2 使用 `--num-gen`

- 只指定 `--num-gen` 时，脚本会保持默认 `batch_size=4` 并自动计算 `num_batches`
- 同时设定 `--batch-size` 或 `--num-batches` 时，另一项会自动调整，总生成数始终大于等于目标
- 例：`--num-gen 100` → `batch_size=4`、`num_batches=25`

命令示例：

```bash
python run_generation.py \
    --results-dir crygen_demo/results/num_gen_100 \
    --model-dir ../mattergen/checkpoints/mattergen_base \
    --num-gen 100
```

运行时如实际生成数与期望不同，会在控制台提示实际 `(batch_size × num_batches)`。

## 3. 远程生成（Bohrium）

### 3.1 提交范例

```bash
python submit_gen.py \
    --dp-email "${DP_EMAIL}" \
    --dp-password "${DP_PASSWORD}" \
    --program-id 29496 \
    --job-name mattergen_generate_demo \
    --remote-model-dir ./model \
    --num-gen 100 \
    --wait
```

关键说明：

- 脚本会创建 `job_<uuid>/` 目录，打包 `run_generation.py` 和指定模型目录一并上传
- 远端用默认镜像 `registry.dp.tech/dptech/dp/native/prod-26745/mattergen-custom:0.0.2`
- 会在容器内先执行 `pip install --no-cache-dir dpdispatcher`，确保依赖可用
- `--wait` 时会在完成后自动拉回 `output/`、`log`、`err` 等文件
- 缺省 `--batch-size`、`--num-batches`、`--num-gen` 时沿用本地默认逻辑

### 3.2 拉取历史结果

作业目录内包含 `pull_results.py`，可在本地执行：

```bash
python crygen_demo/generation/pull_results.py \
    --job-dir crygen_demo/job_xxxxxxxx \
    --dp-email "${DP_EMAIL}" \
    --dp-password "${DP_PASSWORD}" \
    --program-id 29496 \
    --remote-model-path ./model \
    --clean
```

该脚本会重建 dpdispatcher 提交配置，仅下载已完成作业的产物，不会重新提交。

## 4. 本地微调

### 4.1 基本命令

```bash
python finetune/run_finetune.py \
    --data-root crygen_demo/data \
    --model-dir ../mattergen/checkpoints/mattergen_base \
    --output-dir crygen_demo/finetune/outputs/demo \
    --max-epochs 10 \
    --train-batch-size 16 \
    --val-batch-size 16 \
    --devices 1 \
    --accelerator gpu
```

常用参数：

- `--data-root`：需包含 `train/`、`val/` 缓存；可用 `crygen_demo/data` 示例数据
- `--model-dir`：基线模型目录；默认指向官方 `mattergen_base`
- `--output-dir`：Lightning 输出目录
- `--max-epochs`、`--train-batch-size`、`--val-batch-size`
- `--devices` + `--accelerator`：Lightning 设备配置；可设为 `--devices 0 --accelerator cpu`
- `--resume`：继续训练时指定 `checkpoint`
- `--use-wandb`：启用 WandB 记录

脚本内部使用 Hydra 配置（`MODELS_PROJECT_ROOT/conf/finetune.yaml`），默认配置等同于官方脚本。

### 4.2 评估微调效果

```bash
python finetune/evaluate_finetune.py \
    --data-root crygen_demo/data \
    --model-dir ../mattergen/checkpoints/mattergen_base \
    --ckpt crygen_demo/finetune/outputs/demo/checkpoints/last.ckpt
```

会先验证基线模型，再验证指定 checkpoint，并输出指标对比增量。

## 5. 远程微调（Bohrium）

### 5.1 提交命令

```bash
python finetune/submit_finetune.py \
    --dp-email "${DP_EMAIL}" \
    --dp-password "${DP_PASSWORD}" \
    --program-id 29496 \
    --job-name mattergen_finetune_demo \
    --data-root crygen_demo/data \
    --model-dir ../mattergen/checkpoints/mattergen_base \
    --remote-model-dir-name model \
    --max-epochs 5 \
    --wait
```

注意事项：

- 数据与模型默认打包上传到远端 `./finetune_data/` 与 `./model/`
- 若远端已有相同内容，可使用 `--data-root-remote`、`--model-dir-remote` 指定绝对路径，跳过上传
- 结果回传 `job_<uuid>/outputs/`、`lightning_logs/`、`log/` 等目录
- 容器镜像与计算资源参数同生成脚本，可通过 CLI 更改
- `--include-backup` 可保留远端原始目录，便于手动排查

### 5.2 拉取微调产物

```bash
python finetune/pull_results.py --all
```

扫描本地所有 `job_*` 目录，自动解压远程归档，copy 到 `finetune/results/<job_id>/`。

常用参数：

- `job_ids`：指定一个或多个 job（支持 `job_xxx` 或 hash）
- `--dest`：切换目标目录
- `--include-backup`：额外扫描 `job_*/backup/`
- `--dry-run`：仅打印操作
- `--force-extract`：覆盖已有解压目录

## 6. 参数与默认值速查

| 模块 | 参数 | 默认值 | 说明 |
| --- | --- | --- | --- |
| generation.local | `--batch-size` | 4 | 单批次生成数量 |
| | `--num-batches` | 2 | 批次数 |
| | `--num-gen` | 无 | 总生成数（自动规划） |
| generation.remote | `--remote-model-dir` | `./model` | 上传到远端的模型目录 |
| | `--image-name` | `registry.dp.tech/.../mattergen-custom:0.0.2` | Bohrium 镜像 |
| finetune.local | `--data-root` | `crygen_demo/data` | 训练/验证缓存 |
| | `--model-dir` | `../mattergen/checkpoints/mattergen_base` | 基线模型目录 |
| finetune.remote | `--remote-data-dir-name` | `finetune_data` | 数据包上传目录 |
| | `--remote-model-dir-name` | `model` | 模型包上传目录 |

更多细节可在各脚本顶部 docstring 中查看。

## 7. 常见问题

1. **ModuleNotFoundError**：
     - 请确认当前 Python 解释器已安装 `mattergen`、`hydra-core`、`dpdispatcher` 等依赖
     - 入口脚本已自动将 `crygen_demo` 添加到 `sys.path`，无需额外操作
2. **Hydra MissingConfigException**：
     - `--model-dir` 必须是包含 `config.yaml` 的目录，而非单独的 checkpoint 文件
3. **远程任务未回传日志**：
     - 确保提交命令携带 `--wait`；或使用 `pull_results.py` 单独拉取
4. **pip install 日志**：
     - 远程任务在容器内会自动执行 `pip install dpdispatcher`，这是预期行为

## 8. 参考链接

- MatterGen 官方仓库：`../mattergen`
- MatterGen 生成 CLI：`mattergen/scripts/generate.py`
- MatterGen 微调 CLI：`mattergen/scripts/finetune.py`
- dpdispatcher 文档：https://github.com/deepmodeling/dpdispatcher

如需扩展功能，请优先复用 `common/` 中的工具，保持脚本风格一致。
