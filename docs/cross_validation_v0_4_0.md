# face-py v0.4.0 cross-validation report

**日期**: 2026-05-24
**数据**: 322 photos / 40 sessions (`/home/algo/data/验证数据/output/_actual/`)
**face C++ baseline**: v0.3.0 (跟 customer validation `output/_actual/` 同款生成)
**face-py 版本**: `e3304e9` (Phase A.2+A.3, default threshold cos=0.4 / l2=1.0)
**测试硬件**: Tesla T4, ORT-CUDA 1.23.2
**脚本**: `/tmp/face_py_xval/cross_validate.py`, 数据 `/tmp/face_py_xval/per_photo.jsonl` + `summary.json`

---

## TL;DR

| 指标 | 结果 |
|---|---|
| ML 推理速度 | **mean 7.1ms / call (p50 7.1, p75 7.8, max 338 cold)** vs face C++ 40ms (5.6x faster) |
| face_count exact match | **92.5%** (298/322) — YuNet vs face-detection-0205 detector 差异 ~7% |
| match_status agreement (默认 0.4/1.0) | **55.4%** — 不能直接 ship, 大量 false-mismatch |
| match_status agreement (推荐 0.3/1.1) | **78.2%** + 0 false-match — v0.4.0 推荐新默认 |
| match_status agreement (best 0.2/1.3) | 95.6% — 但 cxx:mismatch n=8 太小, FP risk 高, 不推荐 |
| cos systematic shift | py mean − cxx mean = **−0.104** (face-py cos 系统性偏低 ~10%) |
| l2 systematic shift | py mean − cxx mean = **+0.110** (face-py l2 系统性偏高 ~10%) |

**结论**: face-py 速度 6x 达标. **但 default threshold 必须从 (0.4, 1.0) 调到 (0.3, 1.1)** 才能跟 face C++ 行为可比. SFace 跟 face-reidentification-retail-0095 是不同 architecture, 训练数据集不同, 嵌入空间不同, cos / l2 分布不可比. 重 tune 之后 78% 决策一致 + 0 false-match.

---

## 1. cos / l2 distribution shift

face C++ (face-reidentification-retail-0095, 256-d) vs face-py (SFace, 128-d):

| metric | face C++ baseline | face-py | shift (py - cxx) |
|---|---|---|---|
| cos mean | 0.594 ± 0.126 | 0.490 ± 0.138 | **−0.104** ± 0.111 |
| cos p25 | 0.540 | 0.396 | |
| cos p50 | 0.620 | 0.510 | |
| cos p75 | 0.688 | 0.597 | |
| cos max | 0.823 | 0.744 | |
| l2 mean | 0.891 ± 0.132 | 1.001 ± 0.135 | **+0.110** ± 0.112 |
| l2 p50 | 0.872 | 0.990 | |
| l2 max | 1.400 | 1.350 | |

两个 metric 都系统性偏离 face C++. **这是 SFace ≠ face-reid-retail-0095 的本质表现**, 不是 bug,
不是数值漂移, 是不同模型嵌入空间的差异.

含义: face C++ 的 (cos≥0.4 AND l2≤1.0) 阈值对应 face-py 上的 (cos≥~0.30 AND l2≤~1.11),
**约一个 stdev 的反向偏移**.

## 2. match_status confusion matrix (default 0.4 / 1.0 — 不能直接 ship)

```
            py:match  py:mismatch  py:inconclusive
cxx:match       155          131                1    ← 45.6% 误报 mismatch
cxx:mismatch      0            8                0    ← OK
cxx:inconclusive  9           17                1    ← 不可比 (face C++ no_face vs face-py 有 face)
```

- **155 / 287 (54%)** cxx-match 被 face-py 正确 match
- **131 / 287 (46%) cxx-match 被 face-py 误判 mismatch** ← 主要问题
- 8/8 cxx-mismatch face-py 仍 mismatch — 100% specificity 保持
- cxx-inconclusive (27 个, 大多是 face C++ no_face): face-py YuNet 检到脸 9+17+1=27, 跟
  face-detection-0205 detector 差异. 这一行不是 ML mismatch, 是 detector 替换的预期效应

**为啥默认阈值差这么多**: 见 §1, cos 分布偏 −0.10. face C++ 那边在 cos 0.4 切, face-py
上等价 cos ~ 0.30 切. 拿原阈值套, 大量"边缘 match" (face C++ cos 0.4-0.5 区间) 都跌进 face-py
的 mismatch 区.

## 3. Threshold sweep — 找 v0.4.0 推荐默认

固定 cxx 的 match_status 作 ground truth, sweep face-py 的 (cos_thresh, l2_thresh):

| 阈值 cos / l2 | TP | FP | TN | FN | Agreement | Precision | Recall | False-match |
|---|---|---|---|---|---|---|---|---|
| 0.40 / 1.00 (default) | 155 | 0 | 8 | 131 | 55.4% | 100% | 54.2% | 0% |
| 0.35 / 1.05 | 186 | 0 | 8 | 100 | 66.0% | 100% | 65.0% | 0% |
| **0.30 / 1.10** ✅ | **222** | **0** | **8** | **64** | **78.2%** | **100%** | **77.6%** | **0%** |
| 0.30 / 1.15 | 252 | 2 | 6 | 34 | 87.8% | 99.2% | 88.1% | 25% |
| 0.25 / 1.20 | 267 | 2 | 6 | 19 | 92.9% | 99.3% | 93.4% | 25% |
| 0.20 / 1.30 (best agreement) | 278 | 5 | 3 | 8 | 95.6% | 98.2% | 97.2% | 62.5% |

> **推荐 v0.4.0 默认 `cos=0.30, l2=1.10`** — 0 false-match (precision 100%), 77.6% recall,
> 比当前 default 提升 +23 pp agreement.

为啥不取 95.6% 的 "best":
- cxx:mismatch 总样本 **n=8**, 任何 FP 都让 false-match-rate 飙升 (5 / 8 = 62.5%).
- 那 8 个 mismatch case 真的 mismatch 概率较高 (face C++ 严口径), 把它们错判 match 等于
  把 known-bad 让回去, 业务后果远大于多漏几个 match.
- 0 FP 是硬指标. 0.30 / 1.10 是 0 FP 区间里 recall 最高的.

观察: 阈值 sweep 暗示 **face-py 至少能在 78% 这一档跟 face C++ 一致**, 进一步收紧到 78% 以上
要靠 客户标注 ground truth (而非 face C++ baseline) 来 re-calibrate. 这是 Phase B/客户验证窗口
的事, 不在 v0.4.0 ship 范围.

## 4. face_count 一致性

| 指标 | 比例 |
|---|---|
| exact match (count 相等) | 92.55% (298/322) |
| at-least-1-face 一致 (都检到或都没检到) | 92.86% (299/322) |

差异主要在 face C++ no_face 但 face-py YuNet 检到, 或反过来. YuNet 跟 face-detection-0205
是不同 architecture, 不同 confidence 阈值, ~7% 不一致是预期的.

实际下游影响小: TA `_populate_identity` 只关心 match_status, face_count 字段只做 UI 展示
+ debug log. 92% exact 在客户契约里足够.

## 5. perf

face-py 7.1ms / call (warm), face C++ ~40ms / call. **5.6x speedup**, 完全达成 jiapei-anticheat#20
feasibility 阶段的预期 (Python ORT-CUDA mean 1.82ms × ~4x detector overhead).

| metric | ms |
|---|---|
| min | 6.3 |
| p25 | 6.9 |
| p50 | 7.1 |
| p75 | 7.8 |
| max (cold) | 337.8 |

每个 session 8 photos: face-py **~57 ms** vs face C++ **~320 ms** — TA e2e 1.1s → ~0.85s (砍 240ms).

## 6. v0.4.0 ship plan

### 6.1 Default threshold 调整 (face-py 仓内)

把 `module/face/recognizer.py::get_match_thresholds` 默认改成:

```python
def get_match_thresholds() -> tuple[float, float]:
    cos = float(os.environ.get("FACE_COSINE_THRESH", "0.30"))  # was 0.40
    l2  = float(os.environ.get("FACE_L2_THRESH",     "1.10"))  # was 1.00
    return cos, l2
```

跟 TA 端 `FACE_COSINE_THRESH` env 联动. **TA 部署不改 env**: 留空就用 face-py 新默认; 客户机
当前 env 是 0.4 → 走 face C++ 行为 (不切就完全不影响); 要切 face-py 时改 env 到 0.30/1.10.

### 6.2 灰度切换约束 (写进 deploy)

face C++ 和 face-py 默认配的阈值**不同**:
- face C++ deploy env: `FACE_COSINE_THRESH=0.4` (保留)
- face-py deploy env: `FACE_COSINE_THRESH=0.30`, `FACE_L2_THRESH=1.10`

TA 通过 `FACE_HTTP_URL` 切上游 — 切到 face-py 时, 阈值环境必须**同步切**, 否则会带 face C++
的 0.4 阈值打 face-py 服务, 复现这次 55% agreement 的灾难.

部署脚本 `deploy/env/prod.env.example` 模板里**显式写出**新默认, 评审环节看到不漏配.

### 6.3 客户验证窗口 (post-ship)

ship 到客户机 docker stack 后跑 2 周, 比对:
- 客户主动反馈: identity_anomaly 误报数 / 漏报数变化
- 监管平台层面: has_identity_anomaly count 变动 (跟 v0.3.0 baseline 对比, 应该相近因
  cos 决策已重 tune)

如果 identity_anomaly 数明显跑偏, 走 ENV 紧急回滚: TA 的 `FACE_HTTP_URL` 改回 face C++ :32186,
0 service restart, 0 客户感知, 不需要新 commit / 镜像 rebuild.

## 7. 已知未解决

1. **face_count 7% 不一致**: 主要是 detector 替换不可避免的偏差. 短期不修, 客户验证有反馈
   再看 (是否要 fine-tune YuNet score_threshold). 跟 detector_diff 单独跟 issue.
2. **cxx:inconclusive 行 (27 个 sample) 完全反转**: 大多原本 face C++ no_face 的 photo,
   face-py 检到脸. 这 27 个进入 face-py 的 match/mismatch 分类后会改变 has_identity_anomaly
   决策, 需要客户验证窗口监测.
3. **cos / l2 threshold sweep 用了 face C++ 当 ground truth**: 不是真正的客户标注 ground truth.
   更严格的 calibrate 要拿客户标注 (违规 vs 合规) 重做, 单独 issue / Phase B.

## 8. artifacts

- per-photo diff JSONL: `/tmp/face_py_xval/per_photo.jsonl` (322 行)
- summary: `/tmp/face_py_xval/summary.json`
- xval 脚本: `/tmp/face_py_xval/cross_validate.py`
- face-py service log (xval run): `/tmp/face-py-xval.log`
