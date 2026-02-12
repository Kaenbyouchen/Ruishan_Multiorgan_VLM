# VLM-gemini 使用手册

这个项目用于处理 AMOS MRI（3D NIfTI），生成多器官切片可视化，并调用 VLM 输出：

- `hard_classes`（最难分割器官 Top-5）
- `descriptions`（器官形态/大小/位置文本）

同时提供一套完整的检查与评估脚本（缺失检查、GT 对齐检查、规则法对比等）。

---

## 1. 快速上手（3 步）

### 1.1 安装依赖

```bash
pip install -r requirements.txt
```

### 1.2 配置 API Key（真实推理时）

```bash
export OPENAI_API_KEY="your_openai_key"
```

### 1.3 一步到位运行主流程（只跑一个 YAML）

你只需要运行一个 YAML，就会自动完成这条链路：

1. 自动检查/计算每个分箱的代表切片（`average_slice`）  
2. 调用 VLM 输出 `hard_classes` + `descriptions`  
3. 自动生成 `each_class_3words`（把 description 转为三词格式）
4. 自动执行补漏（扫描缺失并重跑缺失 case）

先选 YAML：

- **要输出全部器官的 description**：  
  `configs/pipeline_ten_percent_eachclass_prior_case.yaml`
- **只保留 hard_classes 对应 description**：  
  `configs/pipeline_ten_percent_eachclass_prior_case_harddesc.yaml`

每次运行前建议先检查这几项：

- `dataset.use_all_cases`：`true` 跑全量，`false` 跑单 case
- `dataset.single_id`：单 case 时填写（如 `"507"`）
- `pipeline.output_dir`：本次输出目录（不存在会自动创建）
- `vlm.prompt_template_each_class`：单器官描述提示词
- `vlm.prompt_template_hard_class`：hard class 排序提示词

运行命令（二选一，直接复制其中一条）：

```bash
python src/run_pipeline.py \
  --config configs/pipeline_ten_percent_eachclass_prior_case.yaml

# 或
python src/run_pipeline.py \
  --config configs/pipeline_ten_percent_eachclass_prior_case_harddesc.yaml
```

跑完后主要输出在 `pipeline.output_dir` 下：

- `manifest.json`: 全流程记录（每个 case 的状态、分箱、错误信息等）
- `each_class/*.json`: VLM 原始结构化输出（hard_classes + descriptions）
- `each_class_3words/*.json`: 三词版 description 输出（自动生成）
- `case_<id>/...`: 每个 case 的 raw、overlay、prompt 预览等中间文件
- `vlm_inputs_all/*.png`: 汇总的输入图像（仅当 `save_vlm_inputs_all: true` 时生成）

---

## 2. 目录说明

- `src/`: 所有脚本（主流程 + 检查 + 评估）
- `configs/`: YAML 配置
- `data/`: 输入数据（MRI、pred、gt）
- `outputs/`: 所有输出结果（包括vlm输出json，和各种检查结果）

常用数据目录：

- `data/img_amos_mr`: MRI 影像（`.nii.gz`）
- `data/pred_amos_mr`: 预测 mask（`.nii.gz`）
- `data/gt_amos_mr`: GT mask（`.nii.gz`）

---

## 3. 主配置文件说明（`configs/pipeline_ten_percent_eachclass_prior_case.yaml`）

这个 YAML 决定了主流程几乎所有行为。最常改的是以下参数：

### 3.1 `dataset`（数据范围）

- `image_dir`: MRI 输入目录
- `label_dir`: 分割标签目录（通常是 `pred_amos_mr`）
- `list_file`: case 列表文件
- `ids`: 指定 case 列表（有值时优先，没有也能跑）
- `use_all_cases`: 是否跑全部 case （选true就是跑全部60个case；选false跑下行单个case用于调试）
- `single_id`: 单 case 模式下的 case id

单 case 示例（跑 507）：

```yaml
dataset:
  ids: []
  use_all_cases: false
  single_id: "507"
```

### 3.2 `pipeline`（切片与输出）

- `output_dir`: 输出根目录（输出的所有json都放在这个文件夹下，不存在就自动创建）
- `keep_output_dir`: 是否保留输出目录名，不自动加前缀（true就可以）
- `plane`: `axial` / `coronal` / `sagittal` （咱们水平面就选axial）
- `positions_percent`: 手动百分位切片（空着就行）
- `positions_from_json`: 代表切片 JSON 模板（支持 `{case_id}`）（寻找器官面积中位数json，得到切片位置）
- `positions_json_key`: 读取字段（常用 `average_slice`）
- `auto_compute_positions`: 缺少位置 JSON 时是否自动计算
- `positions_compute_gt_dir`: 自动计算时的 GT 目录
- `positions_compute_output_dir`: 自动计算输出目录
- `positions_compute_window_size`: 百分位窗口大小（每10%找一张slice）
- `run_per_bin`: 是否每个分箱输出一个 JSON
- `use_all_slices`: 是否用全部切片
- `save_vlm_inputs_all`: 是否额外保存 `vlm_inputs_all`（占空间大，不需要时建议 `false`）
- `auto_generate_each_class_3words`: 主流程结束后自动生成 `each_class_3words`
- `auto_fill_missing_cases_after_run`: 主流程结束后是否自动调用补漏
- `auto_fill_max_rounds`: 自动补漏最大轮数
- `auto_fill_sleep_seconds`: 自动补漏时每个 case 的间隔秒数
- `overlay_alpha`: 叠加透明度 （alpha越高mask部分越红越显著）

### 3.3 `vlm`（模型调用）

- `name`: 模型名标识
- `mode`: `infer`（真实推理）/ `mock`（不调用 API）
- `api_key_env`: 默认 API key 环境变量名
- `api_key`: 直接写 key（不推荐）
- `api_key_env_map`: 按模型名映射 key 变量
- `provider`: `openai` 或 `gemini`
- `model`: 实际模型名
- `api_base`: API 地址
- `max_images`: 单请求最大图片数（0 表示不限制）
- `per_class`: 是否每个器官单独出描述
- `require_nonempty_hard_classes`: hard class 是否要求非空描述
- `save_each_class_prompt_preview_first_bin`: 是否保存首个分箱的 prompt 预览
- `force_nonempty_descriptions_for_hard_classes`: 是否强制补非空描述
- `descriptions_only_for_hard_classes`: 是否只保留 hard_classes 的 descriptions（其余器官不输出）
- `prompt_template_each_class`: 单器官描述提示词模板
- `prompt_template_hard_class`: hard class list提示词模板

---

## 4. 所有脚本说明（功能 + 命令 + 可调参数）

下面按 `src/` 全部 11 个脚本逐个说明。

### 4.1 `src/run_pipeline.py`（主入口）

**功能**
- 读取 YAML 配置
- 生成 raw + mask overlay 图像
- 调用 OpenAI/Gemini（或 mock）
- 输出每个 case / 每个分箱 JSON 结果

**命令**

```bash
python src/run_pipeline.py --config configs/pipeline_ten_percent_eachclass_prior_case.yaml
```

**参数**
- `--config`: YAML 配置路径（必填）

---

### 4.2 `src/fill_missing_cases.py`（扫描并补齐缺失输出）

**功能**
- 根据配置推断“应该有的输出”
- 扫描当前缺失项
- 可自动重跑缺失 case（支持多轮重试）
- 说明：当前主 YAML 默认已启用“主流程结束后自动补漏”，本脚本主要用于手动补跑或单独扫描

**命令**

仅扫描：

```bash
python src/fill_missing_cases.py \
  --config configs/pipeline_ten_percent_eachclass_prior_case.yaml
```

扫描并补齐：

```bash
python src/fill_missing_cases.py \
  --config configs/pipeline_ten_percent_eachclass_prior_case.yaml \
  --run \
  --max-rounds 5 \
  --sleep-seconds 2
```

**参数**
- `--config`（必填）: pipeline YAML
- `--run`: 执行补齐（不加则只扫描）
- `--report`: 输出扫描报告 JSON
- `--case-ids`: 仅处理指定 case（逗号分隔）
- `--max-rounds`: 最大重试轮数
- `--sleep-seconds`: case 之间等待秒数

---

### 4.3 `src/convert_each_class_to_3words.py`（描述转 3 词）

**功能**
- 把 `each_class/*.json` 中每个器官描述规范化为 3 段词组
- 输出到同级 `each_class_3words/`

**命令**

```bash
python src/convert_each_class_to_3words.py \
  --each_class_dir outputs/ten_percent_eachclass_prior_case/all_case/each_class
```

**参数**
- `--output_dir`: 上层输出目录（其下必须有 `each_class`）
- `--each_class_dir`: 直接指定 `each_class` 目录

---

### 4.4 `src/check_hardclass_in_gt.py`（hard class 与 GT 对齐检查）

**功能**
- 逐个读取 VLM JSON 的 `hard_classes`
- 检查这些器官是否在 GT 对应切片范围出现
- 输出详细报告（含 summary、按器官统计、逐文件明细）
- 已输出在了outputs/check_hardclass/hardclass_in_gt_check_each_class_3words.json

**命令**

```bash
python src/check_hardclass_in_gt.py \
  --vlm_dir outputs/ten_percent_eachclass_prior_case/all_case/each_class_3words \
  --gt_dir data/gt_amos_mr \
  --plane axial \
  --output_dir outputs/check_hardclass \
  --output_name hardclass_in_gt_check_each_class_3words.json
```

**参数**
- `--vlm_dir`: 待检查 JSON 目录
- `--gt_dir`: GT NIfTI 目录
- `--plane`: `axial|coronal|sagittal`
- `--output_dir`: 输出目录
- `--output_name`: 输出文件名

---

### 4.5 `src/check_pred_masks.py`（pred vs gt 缺失检查）

**功能**
- 检查每个器官在 GT 有但 pred 缺失的切片
- 输出 summary/matrix/dice JSON，可选热力图 PNG
- 已输出在了 outputs/check_pred

**命令**

```bash
python src/check_pred_masks.py --write_heatmap
```

**参数**
- `--gt_dir`（默认 `data/gt_amos_mr`）
- `--pred_dir`（默认 `data/pred_amos_mr`）
- `--list_file`（默认 `data/img_amos_mr/list/dataset.yaml`）
- `--output_dir`（默认 `outputs/check_pred`）
- `--write_heatmap`（是否输出热力图）

---

### 4.6 `src/organ_distribution.py`（器官出现比例分布）

**功能**
- 统计 0%-100% 位置上器官“是否出现”的病例占比
- 输出单器官曲线、全器官曲线、峰值区间 JSON
- 已输出在了 outputs/organ_distribution

**命令**

```bash
python src/organ_distribution.py \
  --gt_dir data/gt_amos_mr \
  --pred_dir data/pred_amos_mr \
  --plane axial
```

**参数**
- `--gt_dir`（必填）
- `--pred_dir`（必填）
- `--output_dir`（默认 `outputs/organ_distribution`）
- `--plane`（默认 `axial`）
- `--window_size`（默认 `10`）

---

### 4.7 `src/organ_square_distribution.py`（器官面积分布与代表切片）

**功能**
- 对每个 case 统计各器官面积占自身最大面积的百分比
- 输出每个分箱的代表位置/代表切片
- 生成 `pred_representative_positions.json` 给主流程使用
- 已输出在了 outputs/organ_square_distribution

**命令**

```bash
python src/organ_square_distribution.py \
  --gt_dir data/gt_amos_mr \
  --pred_dir data/pred_amos_mr \
  --case_id 507 \
  --plane axial
```

**参数**
- `--gt_dir` / `--pred_dir`: GT 与 Pred 目录（两个都要）
- `--case_id`: 指定单 case（不填则全量）
- `--output_dir`（默认 `outputs/organ_square_distribution`）
- `--plane`（默认 `axial`）
- `--window_size`（默认 `10`）

---

### 4.8 （已废弃不需要）`src/run_heuristic_pipeline.py`（规则法基线）

**功能**
- 不调用 VLM，直接用 mask 几何规则生成 hard class + descriptions
- 适合做基线或对照

**命令**

```bash
python src/run_heuristic_pipeline.py --config configs/pipeline_ten_percent_eachclass_prior_case.yaml
```

**参数**
- `--config`: 配置文件（必填）

---

### 4.9 （已废弃不需要）`src/eval_vlm_vs_heuristic.py`（VLM 与规则法对比）

**功能**
- 对齐同名 JSON 文件
- 评估 hard class 重叠与描述字段匹配率

**命令**

```bash
python src/eval_vlm_vs_heuristic.py \
  --heuristic_dir outputs/ten_percent_eachclass_prior_case_507/heuristic_outputs \
  --vlm_dir outputs/ten_percent_eachclass_prior_case_507/each_class \
  --output outputs/vlm_vs_heuristic.json
```

**参数**
- `--heuristic_dir`: 规则法目录（必填）
- `--vlm_dir`: VLM 输出目录（必填）
- `--output`: 输出 JSON 路径（必填）

---

### 4.10 （已废弃不需要）`src/compare_vlm_vs_square_hardclass.py`（VLM vs 面积排序 hard class）

**功能**
- 用 `pred_summary.json` 的面积信息构造 hard class 排序
- 与 VLM 的 hard class 按 bin 对比
- 输出 overlap / set match / ordered match 等指标

**命令**

```bash
python src/compare_vlm_vs_square_hardclass.py \
  --case_id 507 \
  --vlm_output_dir outputs/ten_percent_eachclass_prior_case_507/each_class \
  --positions_dir outputs/organ_square_distribution \
  --output_dir outputs/hardclass_compare
```

**参数**
- `--case_id`: 指定 case（不填则扫全部）
- `--vlm_output_dir`: VLM JSON 目录
- `--pred_dir`: 兼容保留参数（当前实现未使用）
- `--positions_dir`: `organ_square_distribution` 输出目录
- `--output_dir`: 报告输出目录

---

### 4.11 `src/rename_gt_masks.py`（GT 文件名批量重命名）

**功能**
- 把 `*_gt.nii.gz` 改成 `*.nii.gz`

**命令**

先预览：

```bash
python src/rename_gt_masks.py --gt_dir data/gt_amos_mr --dry_run
```

再执行：

```bash
python src/rename_gt_masks.py --gt_dir data/gt_amos_mr
```

**参数**
- `--gt_dir`: GT 目录（必填）
- `--dry_run`: 仅打印不改名

---

## 5. 常见运行场景

### 5.1 跑单 case / 跑 all case

单 case（例如 507）在 YAML 里设置：

```yaml
dataset:
  ids: []
  use_all_cases: false
  single_id: "507"
```

然后运行：

```bash
python src/run_pipeline.py --config configs/pipeline_ten_percent_eachclass_prior_case.yaml
```

all case 在 YAML 里设置：

```yaml
dataset:
  ids: []
  use_all_cases: true
  single_id: ""
```

如果你希望输出自动落在 `.../all_case` 下，建议同时设置：

```yaml
pipeline:
  keep_output_dir: false
```

然后运行同一条命令：

```bash
python src/run_pipeline.py --config configs/pipeline_ten_percent_eachclass_prior_case.yaml
```

### 5.2 不调用 API 先测流程

把 YAML 改成：

```yaml
vlm:
  mode: mock
```

### 5.3 输出缺失后一键补齐

```bash
python src/fill_missing_cases.py \
  --config configs/pipeline_ten_percent_eachclass_prior_case.yaml \
  --run \
  --max-rounds 5 \
  --sleep-seconds 2
```

---

## 6. 产出文件速查

- 主流程：
  - `outputs/<your_output_dir>/manifest.json`
  - `outputs/<your_output_dir>/each_class/*.json`（per-class）
  - `outputs/<your_output_dir>/vlm_outputs/*.json`（非 per-class）
- 转三词：
  - `outputs/.../each_class_3words/*.json`
- GT 检查：
  - `outputs/check_hardclass/*.json`
- pred 检查：
  - `outputs/check_pred/*.json`

---

## 7. 注意事项

- 建议 API key 只用环境变量，不要写进 YAML。
- `infer` 模式下，API 额度/限流会影响完成速度与成功率。
- 如遇中断，优先用 `fill_missing_cases.py` 补齐，不要手工逐个重跑。