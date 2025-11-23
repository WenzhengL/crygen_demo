## fine_gen 流水线说明（微调 → 评估 → 生成 → 评估生成结构）

本 README 说明 `crygen_demo/fine_gen` 目录下的一键流水线脚本，以及如何在本地和远程 (Bohrium) 环境中使用它完成：

1. 使用指定数据对 MatterGen 进行 **微调**；
2. 对比 **基线模型 vs 微调模型** 的验证指标；
3. 使用微调后的模型 **生成晶体结构**；
4. 调用 MatterGen 的评估工具，对生成的结构进行 **结构稳定性/新颖性等指标评估**。

> 约定：重依赖（`mattersim`、完整的 MatterGen 依赖）只在远程容器里安装并使用，本地环境保持轻量，只负责提交任务和收集结果。

---

### 目录结构概览

- `crygen_demo/fine_gen/run_fine_gen.py`
  - 本地/远程均使用的核心流水线脚本：**微调 → 评估 (finetune) → 生成 → 评估生成结构**，并写出 `pipeline_run/summary.json`。
- `crygen_demo/fine_gen/remote.py`
  - 使用 `dpdispatcher` 将 `run_fine_gen.py` 提交到 Bohrium 运行，自动打包 `common/`, `finetune/`, `generation/`, `fine_gen/` 以及 `mattergen/` 源码。
- `crygen_demo/fine_gen/submit_fine_gen.py`
  - 简单的 CLI 包装器，从当前工程根目录调用，负责修正 `sys.path` 并转发参数到 `fine_gen.remote`。
- `crygen_demo/generation/evaluate.py`
  - 对 `generated_crystals.extxyz` 中的生成结构调用 `mattergen.evaluation.evaluate` 计算指标，输出 `metrics.json` 与 `metrics_summary.json`。
- `crygen_demo/generation/remote_eval.py` + `submit_eval_gen.py`
  - 若需要仅对某一次生成结果做单独评估，可用这两个脚本把评估任务提交到 Bohrium。

---

### 一、远程端到端流水线：微调 + 评估 + 生成 + 评估生成结构

#### 1. 远程运行环境假设

- 镜像中已有 `/root/dev/mattergen/.venv`，其中包含 PyTorch、`mattersim` 等完整依赖。
- 镜像内可以通过 `pip install ./mattergen` 安装本仓库的 `mattergen` 包，提供 `mattergen-generate` 等 CLI。

`fine_gen/remote.py` 中的远程命令大致为：

1. 激活虚拟环境：`source /root/dev/mattergen/.venv/bin/activate`；
2. 安装 `dpdispatcher`；
3. 解压 `support.tar.gz`，得到 `common/`, `finetune/`, `generation/`, `fine_gen/`, `mattergen/` 源码；
4. 执行 `pip install ./mattergen`；
5. 运行 `python run_fine_gen.py ...`。

#### 2. 提交远程 fine_gen 流水线

在仓库根目录下执行（参数请替换成你的账户信息）：

```bash
python crygen_demo/fine_gen/submit_fine_gen.py --wait \
  --dp-email <your_email> \
  --dp-password '<your_password>' \
  --program-id 29496 \
  --max-epochs 1 \
  --num-gen 4
``

主要参数含义：

- `--max-epochs`：微调 epoch 数（demo 可用 1）；
- `--num-gen`：使用微调模型生成的结构数量；
- 其余关于资源规格 (`--scass-type`, `--platform`, `--image-name`) 有默认值，一般保持不变即可。

任务完成后，本地会出现一个新的 job 目录，例如：

- `crygen_demo/fine_gen/job_xxxxxxxx/pipeline_run/`
  - `finetune_outputs/checkpoints/last.ckpt`：微调后的模型；
  - `evaluation/metrics.json`：基线与微调模型在验证集上的指标对比；
  - `generation/model_for_generation/`：用于生成的模型目录（包含基线 config + 微调 checkpoint）；
  - `generation/results/`：生成的结构文件：
    - `generated_crystals.extxyz`
    - `generated_crystals_cif.zip`
    - `generated_trajectories.zip`
  - `summary.json`：整个流水线的汇总信息（见下一节）。

---

### 二、`run_fine_gen.py` 的行为与 `summary.json` 内容

`run_fine_gen.py` 主要包含四个步骤：

1. `_finetune(...)`：调用 `finetune.local.run_finetune`，以 `--data-root` 和 `--base-model-dir` 为输入，在 `pipeline_root/finetune_outputs` 下进行微调，并强制写出 `checkpoints/last.ckpt`。
2. `_evaluate(...)`：
   - 使用微调前的基线模型和微调后的 checkpoint，在同一验证集上运行 Lightning `validate`；
   - 输出 `evaluation/metrics.json`，并返回一个字典：
     - `baseline`: 各种 loss / metric 数值；
     - `finetuned`: 同一指标在微调后模型上的数值；
     - `delta`: `finetuned - baseline` 的差值。
3. `_generate(...)`：
   - 构造 `generation/model_for_generation/`：
     - 先将 `base_model_dir` 下的 config / 资源复制过去（排除原始 `.ckpt`）；
     - 再把微调得到的 `last.ckpt` 拷贝为 `model_for_generation/last.ckpt`；
   - 目标结构数 = `num_gen`；根据此值自动选择 batch size 和 batch 数；
   - 优先调用 CLI：
     ```bash
     mattergen-generate <results_dir> \
       --model_path=<model_for_generation> \
       --batch_size=<bs> --num_batches=<nb>
     ```
   - 如果 CLI 不可用或失败，则退回到 Python API：
     `mattergen.scripts.generate.main(...)`，并指定 `sampling_conf`；
   - 生成的结构写入 `generation/results/`。若全部失败，会创建 `GENERATION_FAILED.txt` 标记原因。
4. `_evaluate_generated_structures(gen_dir)`：
   - 入口 `gen_dir = generation/results/`；
   - 若目录中已存在 `metrics_summary.json`，直接读取；
   - 否则尝试导入 `generation.evaluate` 并直接调用其 `run()`：
     - 这里**依赖远程容器已经安装 `mattergen` 全依赖**（包括 `mattersim`）。
     - 评估结果写回同一个 `gen_dir` 中的 `metrics.json` 与 `metrics_summary.json`。
   - 若评估脚本/依赖不可用，记录一个带有 `error` 字段的字典，避免整个流水线失败。

最终 `summary.json` 结构大致为：

```json
{
  "checkpoint": "<path-to-last.ckpt>",
  "metrics": { "baseline": {...}, "finetuned": {...}, "delta": {...} },
  "generation_dir": "<path-to-generation/results>",
  "generation_eval": {
    "count": <生成结构数>,
    "metrics": { ... },
    "relaxed": false
  }
}
```

如果远程评估阶段缺失参考数据集（例如 `reference_MP2020correction.gz` 未被正确安装或可见），`generation_eval` 中会包含：

```json
"generation_eval": {
  "error": "evaluation failed: ..."
}
```

这有助于调试远程镜像中 MatterGen 数据发布包 (`data-release/`) 的挂载问题，而不会影响流水线整体成功完成。

---

### 三、单独评估已有生成结果（只在远程环境跑评估）

如果你已经用 `fine_gen` 或手动 `generation/local.py` 得到了一份生成结果目录（包含 `generated_crystals.extxyz`），可以单独发起一次“只评估生成结构”的远程任务，而不重新微调/生成：

```bash
python crygen_demo/generation/submit_eval_gen.py -- \
  --wait \
  --dp-email <your_email> \
  --dp-password '<your_password>' \
  --program-id 29496 \
  --results-dir crygen_demo/fine_gen/job_XXXXXXXX/pipeline_run/generation/results \
  --no-relax
```

说明：

- `submit_eval_gen.py` 会调用 `generation.remote_eval`：
  - 打包 `generation/` + `common/` + `mattergen/` 源码；
  - 在远程容器中安装 `dpdispatcher` + `./mattergen`；
  - 解压你指定的 `results-dir` 内容到远程；
  - 调用 `python -m generation.evaluate --results-dir results --no-relax`；
  - 将评估后的结果（包括 `metrics.json`, `metrics_summary.json`）回传到一个新的本地 job 目录：
    - `crygen_demo/generation/job_xxxxxxxx/results/`。

你可以直接查看：

- `crygen_demo/generation/job_xxxxxxxx/results/metrics_summary.json`

了解该批生成结构的稳定性、新颖性等指标表现。

---

### 四、建议的使用路径

1. **快速端到端试跑**：
   - 直接用 `submit_fine_gen.py` 跑一个 epoch + 少量 `num_gen`，验证流水线结构是否正常；
   - 查看 `fine_gen/job_*/pipeline_run/summary.json` 中的两类指标：
     - 微调前后验证集指标 (`metrics`)
     - 生成结构评估结果 (`generation_eval`)；
2. **正式实验**：
   - 适当增大 `--max-epochs`、`--num-gen`，保持其它配置不变；
   - 若只想针对某次生成结果反复调评估参数（例如是否 relax、使用不同 potential），可通过 `submit_eval_gen.py` 单独发起评估 job。

如需进一步把 `summary.json` 中的信息汇总成表格、画图，建议在 `iter_0/eval` 或单独的 notebook 中读取 `summary.json` / `metrics_summary.json` 做分析，这部分不依赖远程环境，可以完全在本地完成。
