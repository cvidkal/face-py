# HANDOFF — Phase A.7: 两阶段 identity verification

**上一 session 结束时间**: 2026-05-24
**当前 face-py 版本**: 0.0.1, main HEAD `4138e1f`
**下一 session 任务**: 实现两阶段 session-level identity 判断, 取代单 photo 直接判.

---

## 一句话 context

face-py 现在 per-photo 判 match/mismatch (quality gates + cos tri-state), 但 user 在
session 探讨末尾给了**更好的 framing**: 人脑判代训分两步走 — (1) 先看 session 内部
N 张 photo 互相是不是同一个人, (2) 再看那个人跟 ref 像不像. 当前 face-py 只做 (2)
变体, 漏 (1). rec#2 真造假 (sign_out 是替考) 就被 (1) 直接抓住.

**下一 session 实现 (1)+(2) 两阶段, 在 322 photo + 客户 GT 上验证比 A.5 单 photo
路径更好.**

---

## 必读上下文 (按重要性排)

### 1. user 给的 framing — 一定要看

session 末尾对话, user 反驳了我之前 "代训影响整个 session" 的假设. 实际是:

> "人来看待一个 session 是否代训, 只要看两个. 一个是 session 内部稳不稳定, 我先
> 不管这个人是谁, 内部一串图片是不是同一个人, 不会男的变女的, 老的变小的, 白人
> 变黑人这样子, 也就是 identity 要一致. 然后我们再来谈这个人是不是 reference
> 那个人, 那如果人去判断的时候, 如果其中一张不好确认的话, 那可能会找另外一张
> 看, 找视角好的看."

key insight:
- **Stage 1 (内部一致性)**: pairwise cos 在所有 session photo 之间. 同一人 = pairwise cos 高;
  代训 (尤其 split-session 像 rec#2) = 有 outlier photo
- **Stage 2 (vs ref)**: 只在 Stage 1 通过后跑. 用 session 高质量 photo 集合的 "session
  prototype" embedding (median 或 mean) vs ref, 不是单 photo
- "找视角好的看" = quality gate 已经在做 (Phase A.5)

### 2. 现有 face-py 状态 (Phase A.5 + A.6 已 ship)

- 仓: https://github.com/cvidkal/face-py, local `/home/algo/face-py`
- main HEAD `4138e1f` (Phase A.6 ROC analysis 文档)
- 服务在 `/home/algo/face-py/service/serve.py`, port 32192 默认 (跟 face C++ :32186 共存)
- pipeline 是 `module/face/pipeline.py::FacePipeline.identity_check`, 单 photo decision
- quality gates 在 `module/face/quality_gate.py`, 默认 det≥0.88 / bbox≥40 / clarity≥30 / pose
- cos tri-state: cos<0.15 → mismatch, cos≥0.30 + l2≤1.15 → match, 中间 → inconclusive
- 跑过 322 photo + 客户 GT, 在 customer GT 上 recall 100% spec 94.9% (好于 face C++ 92.3%)
- 在 11.5K cross-student pair 上 ROC knee = cos 0.30 (=A.5 default), 已验证最优

### 3. 现有 TA 集成状态

TA 当前 production 仓 (`cvidkal/training_analyzer`):
- Phase 1 / 3 / 4 perf 优化已 ship 到 main, e2e 14s → 1.1s (10x)
- face-py 还**没接入** TA — TA 的 `FACE_HTTP_URL` 仍指 face C++ :32186
- face-py 灰度部署 (Phase B) 没做

TA `_finalize_session_level` 现在的 identity 聚合是脆弱的:
```python
sl.has_identity_anomaly = (sl.mismatched_photos > 0)   # 单 photo 触发
```
这一条是数据级别错位 (一个噪声 photo 拖累整 session), 但**改这个不是 A.7 的工作** —
A.7 是把"内部一致 + vs ref"整套塞进 face-py 一个新 endpoint, TA 这边的 aggregation
直接信 face-py 的 session-level 决策.

### 4. 关键数据 / 文件位置

| 路径 | 内容 |
|---|---|
| `/home/algo/data/验证数据/output/S*/ai_params.json` | 40 个 session 的原始请求 (有 student_id / session_id / photos[]) |
| `/home/algo/data/验证数据/output/_actual/*.actual.json` | face C++ v0.3.0 baseline 响应, ground truth 比对用 |
| `/home/algo/data/验证数据/audit_vs_actual.md` | 客户 41 条 audit, ground truth 来源 — **rec#2 (S177509932310186) 是唯一 known 真造假** |
| `/var/lib/training_analyzer/sessions/_bench/<sid>/<ssid>/` | algo-readable 已 materialize 的 photo (322 张), 跟 ai_params 对得上 |
| `/tmp/face_py_xval/per_photo.jsonl` | Phase A.4/A.5 每张 photo 的 base vs py 决策对照 (322 行) |
| `/tmp/face_py_xval/pair_cos.jsonl` | Phase A.6 全部 11,520 个 pair 的 cos (288 same + 11,232 cross) |
| `/tmp/face_py_xval/cross_validate.py` | 客户 GT 验证脚本, 跑 322 photo 过 face-py 拿对照 |
| `/tmp/face_py_xval/roc_analysis.py` | 抽 embedding + 算 pair-wise cos + ROC, 可复用 |
| `/tmp/face_py_xval/explore_quality.py` | 数据探索, 拿 det_score / bbox / clarity / cos 全分布 |
| `/tmp/face_py_xval/sweep_gates.py` | quality gate 参数 sweep, 可改 evaluate() 接 multi-photo rule |

### 5. ML 模型 + 推理路径

- face-py 模型: `/home/algo/face-py/models/`, 两个 ONNX (YuNet detect + SFace recognize)
- ORT-CUDA 跑在 T4, mean 7.1ms/call (face-py service 端到端)
- 推理 lib: `pip install --user onnxruntime-gpu==1.23.2` 已装
- CUDA 12.8 在 `/usr/local/cuda-12.8/`, cuDNN 9 同路径下 `lib`
- 服务启动: `cd /home/algo/face-py && LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH FACE_HTTP_PORT=33293 python3 service/serve.py`
- 验 GPU 跑通: `curl http://127.0.0.1:33293/api/v1/health` 看 providers 含 CUDAExecutionProvider

---

## A.7 具体要做的事 (按顺序)

### A.7.1 — 数据探索 + 算法原型 (~1 hour, 先 prototype 别动 face-py 源)

写 `/tmp/face_py_xval/two_stage_proto.py`:

1. 从 `/tmp/face_py_xval/pair_cos.jsonl` 提 same-person 内 photo embedding (实际不在 jsonl
   里, 要从头跑 — 但 `roc_analysis.py` 已经把 embedding 抽出来过, 可改成把 embedding
   持久化到 `/tmp/face_py_xval/embeddings.npz` 给后续复用; 直接复用 explore_quality.py
   骨架抽一次就行)

2. Stage 1 算法 — outlier detection:
   - 每 session 的 N photo embedding 做 NxN cos 矩阵
   - 对每张 photo, 算它对其它 photo 的 mean cos (excluding self)
   - 把 photo 按 mean cos 排序: 最低的 = candidate outlier
   - 阈值: candidate 的 mean cos < X (建议起 X=0.20) → mark outlier
   - session 有 outlier → 内部不一致, 直接报代训
   - 注意: 单 photo session 没法做 Stage 1, fall through to Stage 2

3. Stage 2 算法 — session prototype vs ref:
   - 取 Stage 1 的 consistent core (排除 outliers)
   - 计算 core 的 mean embedding (L2 normalize 后)
   - prototype vs ref cos → tri-state classify (沿用 A.5 cos<0.15/cos≥0.30 thresh)

4. 跑 322 photo / 40 session, 对客户 GT 算 session-level confusion matrix:
   - rec#2 (S177509932310186): 期望 Stage 1 catch sign_out outlier → 代训 ✓
   - 其它 39 session clean: 期望 Stage 1 + Stage 2 都 pass → 不报警 ✓
   - 期望最终: recall 100%, FP ≤ 1 (vs A.5 单 photo 的 FP=2)

5. 也跑 cross-student pair benchmark (~1521 cross-session pair):
   - 用学员 A 整个 session 的 prototype vs 学员 B 的 ref
   - 期望 cross-session FP rate << 4.80% (现 ANY rule) 且 << 1.58% (单 photo)

### A.7.2 — 实现 face-py 新 endpoint (~半天)

加 `POST /api/v1/face/session_check`:

请求 shape (跟 TA 现有数据流 align):
```json
{
  "ref_image_path": "/.../ref.jpg",
  "photos": [
    {"sequence_no": 1, "photo_type": "sign_in",  "image_path": "/.../seq_001_sign_in.jpg"},
    {"sequence_no": 2, "photo_type": "process",  "image_path": "/.../seq_002_process.jpg"},
    ...
  ]
}
```

响应:
```json
{
  "session_status": "match" | "mismatch" | "inconclusive",
  "internal_consistency": "consistent" | "inconsistent",
  "outlier_sequence_nos": [11],   // Stage 1 输出
  "session_cos_to_ref": 0.52,     // Stage 2 输出
  "session_l2_to_ref": 0.93,
  "photo_results": [                // 兼容现有 identity_check 字段
    {"sequence_no": 1, "match_status": "match", "cosine_score": 0.62, ...},
    ...
  ],
  "elapsed_ms": ...
}
```

实现要点:
- `module/face/pipeline.py` 加 `session_check(ref_path, photos)` 方法
- 复用 identity_check 的 quality gate + ref cache + per-photo embedding 抽取
- 加 `module/face/session_consistency.py` 跑 Stage 1 outlier detection
- 单 photo / 全 inconclusive 兜底: session_status = "inconclusive"

### A.7.3 — 更新 docs + tests (~半天)

- `docs/cross_validation_v0_4_0.md` §23+ 加 Phase A.7 section
- `module/face/session_consistency.py` 加单元测试 (mock embeddings, 测 outlier detect 边界)
- `tests/test_pipeline.py` (新) 测 session_check end-to-end mock
- 更新 CLAUDE.md "API endpoints" 段加新 endpoint

### A.7.4 — commit + push (~10 分钟)

跟 Phase A.5 / A.6 同款 commit 风格. push to main (不动 TA 仓).

### A.7.5 (optional, 如果时间够) — TA 接 face-py session_check

新 issue, 不在 A.7 范围:
- TA `module/clients/face_client.py` 加 `face_session_check(...)` 调用
- TA orchestrator 改 identity 路径: 不再 8 次单 photo, 改 1 次 session_check
- TA `_finalize_session_level` 信 face-py 给的 session_status, 不再 ANY mismatch rule

---

## 不在 A.7 范围 (避免 scope creep)

- 不动 quality gate 阈值 (Phase A.5 定了, A.6 验过)
- 不动 SFace 模型选型 (Phase B 想做再说)
- 不动 docker / deploy bundle (Phase B)
- 不接 TA 端代码 (A.7.5 单独 issue)

---

## Open questions 留给 user / 下一 session 现场决策

1. **Stage 1 outlier 阈值 X 用什么** — prototype 跑出来才知道. 起 X=0.20 试, 看 ROC 调.
2. **session prototype 用 mean 还是 median embedding** — 都试, 选效果好的 (probably mean 更好, embedding 加性).
3. **outlier 多张时怎么办** — N photo 里 2 张 outlier (i.e. session 一半人 A 一半人 B), 是哪个 cluster 算"主"? 简单: 用 majority cluster 的 mean 作 prototype, outlier cluster ≥ 1 张就报代训.
4. **TA 接入要不要在 A.7 一起做** — 我倾向不做, A.7 就 face-py 单仓 ship + benchmark, TA 集成单独 issue.

---

## Quick start for 下一 session

```bash
# 进 face-py
cd /home/algo/face-py
git log --oneline | head -5
cat docs/HANDOFF_phase_A7.md        # 这个文件

# 验证 face-py service 仍能跑
ps -ef | grep "face-py.*serve" | grep -v grep
# 如果没在跑:
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib python3 service/serve.py &
curl http://127.0.0.1:32192/api/v1/health

# 看上次 cross-validation 数据 (确认 baseline 还在)
ls /tmp/face_py_xval/
head -3 /tmp/face_py_xval/pair_cos.jsonl

# 开干 A.7.1: prototype 脚本
cd /tmp/face_py_xval
# 编辑 two_stage_proto.py (新)
```

跟 user 同步进度: 评论 `cvidkal/training_analyzer#41`.
