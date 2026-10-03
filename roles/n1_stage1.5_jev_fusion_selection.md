# Stage1.5：Jev 融合方案选择

**阶段顺序：[Stage1 需求分析](n1_stage1_requirements_analysis.md) → Stage1.5 融合方案选择 → [Stage2 首版实现](n1_stage2_first_impl.md)。**

当前实现目标为 Triton / triton-ascend（昇腾 NPU）；评分时必须结合该后端的实际表达能力，不把其他框架或 GPU 专用能力视为已具备。

本文件说明本阶段的职责、输入和输出，供人阅读，不直接作为 Jev 请求。实际执行由 `orchestrator.py` 调用 `lib/fusion_selection.py` 完成；发送给 Jev 的内容全部为英文。

## 输入

路径基准在表中注明；程序读取文件并组织英文内容，Jev 不直接访问这些本地路径。

| 相对路径 | 相对目录 | 作用与阅读方式 |
|---|---|---|
| `knowledge/fusion_method.md` | 项目根 | 融合方法原文；理解各方法的数据流、适用条件及限制。 |
| `knowledge/fusion_options.json` | 项目根 | 详细候选目录，含 F1–F10 共 10 类、24 个变体；按固定 ID 比较方法及其约束，不把变体当成额外评分项。 |
| `fusion_requirements.en.json` | `<work>` | Stage1 的融合需求摘要；结合语义、case 特征、精度与实现约束判断适用性。 |
| `device_info.json` | `<work>` | 当前硬件真实参数；用架构、核数及存储容量核对方法前提。 |
| `ANALYSIS.md` | `<work>` | 完整需求分析；程序读取其哈希用于输入一致性检查，正文不直接发送给 Jev，评分使用上述需求摘要。 |

## 职责

1. 程序复用 Kerminal 将非英文材料翻译为英文，保留方案 ID、数值和 JSON 结构，检查译文与请求大小。
2. 程序组织 `model / state / questions` 请求；Jev 对每个大类给出独立的 `noul` 适用性概率。24 个变体作为方案详情参与判断，不单独评分。
3. 程序校验所有方案都有合法概率，按概率降序排列；同分保留目录顺序，取前 n 个组成 **JSON 融合算子库**。

`n` 由 `config.yaml` 的 `fusion_selection.top_n` 配置，默认 3。概率表示“该方向可行且值得探索”的初始判断，不要求总和为 1，不代表实测性能或唯一最优方案。此阶段不生成算子代码或自测报告。

## 输出与路由

| 输出 | 内容与用途 |
|---|---|
| `<work>/fusion/jev_request.json`、`jev_response.json` | 完整英文请求和 Jev 原始响应，供追溯 |
| `<work>/fusion/ranking.json` | 全部 10 个大类的概率与排序 |
| `<work>/fusion/fusion_library.json` | 前 n 个完整候选，每项包含 `rank`、`probability`、`method`，传给 Stage2、3、7、8、9 |

Stage1.5 成功后进入 Stage2，以概率最高的可实现方案作为首版方向；Stage3、7、8、9 结合各自职责和实际评测证据参考方案库。输出库保留原目录的方案详情，翻译仅用于 Jev 输入。

翻译、请求校验或评分失败时停止，不进入 Stage2；输入一致时可复用已校验的结果，只修改 n 时直接重选。本阶段沿用既有后续路由、P0/P1/P2 和退出逻辑。
