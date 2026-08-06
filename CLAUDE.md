# CLAUDE.md — face-py

> Face inference HTTP service (Python + ONNX Runtime + CUDA). 替代 cvidkal/jiapei-anticheat
> 的 C++ face_http_server (除 segment 外).
>
> 上游 issue: cvidkal/training_analyzer#41 (face-py Python 重写).
> Feasibility 数据 + pivot rationale: cvidkal/jiapei-anticheat#20 (closed as superseded).
> Perf 上下文: training_analyzer perf umbrella cvidkal/training_analyzer#39.

## 心智模型 — 是什么 / 不是什么

**是**: 一个 Python HTTP 服务, 走 ONNX Runtime + CUDA 12 推理. 端口 32192 (新, 跟 face C++ 的
32186 分开, 灰度切换期两个都跑). 客户端**只透过 TA**, 不直接打 face-py — TA 的
`FACE_HTTP_URL` 配到 face-py:32192 即生效, 客户契约不变.

**不是**:
- 不是编排服务. 编排在 TA. face-py 只接 image_path 进来, 输出 embedding / cos / match_status / error_code.
- 不是模型训练. 模型是预训练的 ONNX (YuNet detect + SFace recognize), 这里只跑推理.
- 不是 segment 服务. `/face/segment` **不在本仓**, 留给 face C++ 一段时间, 单独 issue 跟踪
  Python 化. 见 "拓扑" 段最后一句.

## 拓扑

```
                    监管平台
                       │
                  ┌────▼──────┐
                  │ TA :32187 │
                  └─┬─────────┘
                    │
                ┌───┼────────────────────────────────────┐
                │   │                                    │
       FACE_HTTP_URL│        FACE_SEGMENT_HTTP_URL      │
                ▼   ▼              ▼                     ▼
        ┌─────────────────┐  ┌─────────────────┐  ┌───────────┐
        │ face-py :32192  │  │ face C++ :32186 │  │ cloth/...  │
        │ (本仓)          │  │ (legacy, segment │  │            │
        │ identity_check  │  │  only)           │  │            │
        │ compare         │  │                 │  │            │
        └─────────────────┘  └─────────────────┘  └───────────┘
```

- TA env `FACE_HTTP_URL` 指 face-py — 主流程 8-photo identity_check 走这, perf 大头.
- TA env `FACE_SEGMENT_HTTP_URL` 暂时指 face C++ — segment 没在主流程, 流量低,
  port 到 Python 是 Phase E+ 的独立 issue.

## API endpoints

跟 face C++ byte-level 兼容. 字段措辞 / shape / error_code 全保留, 客户感知零变化.

### `POST /api/v1/face/identity_check`

主流程 8-photo 都打这个. 详见 `module/face/pipeline.py::identity_check`.

请求:
```json
{ "image_path": "/var/lib/.../seq_001_sign_in.jpg",
  "ref_image_path": "/var/lib/.../ref.jpg" }
```

响应 (成功):
```json
{ "status": "ok",
  "face_count": 1,
  "cosine_score": 0.5219,
  "l2_distance": 0.9777,
  "clarity_score": null,
  "match_status": "match",
  "elapsed_ms": 5.4 }
```

响应 (per-photo 错误路径):
- `no_face` — YuNet 没检到任何 face
- `image_read_failed` / `ref_image_read_failed` — opencv 读图失败
- `feature_extraction_failed` — embedding 抽取异常 (catch-all)
- `pose_excessive` — yaw > 0.35 OR pitch > 0.55 (跟 face C++ #13 patch 后默认对齐)
- `detection_low_confidence` (**Phase A.5 新**) — YuNet 自己 score < 0.88, 检测不可信
- `face_too_small` (**A.5 新**) — bbox min(w,h) < 40px, SFace 输入向上 upscale 不稳
- `face_too_blurry` (**A.5 新**) — aligned 112×112 crop Laplacian variance < 30
- `cos_inconclusive_zone` (**A.5 新**) — cos ∈ [0.15, 0.30) 中间区, 拿不准就不报

**决策三态** (`match_status`):
```
cos < FACE_COSINE_MISMATCH_THRESH (默认 0.15)        → mismatch  (高置信不像)
cos ≥ FACE_COSINE_THRESH (默认 0.30) AND l2 ≤ 1.15  → match     (高置信像)
else                                                  → inconclusive
```

阈值默认值跟 face C++ **不通用**:
- face C++ 默认 cos 0.4 / l2 1.0 (face-reid-retail-0095, 256-d)
- face-py 默认 cos 0.30 / l2 1.15 (SFace 128-d, cos 系统性偏低 ~0.10)

切到 face-py 时 TA 端**不需要**改 FACE_COSINE_THRESH — TA 只看 has_identity_anomaly,
来自 face-py 的 match_status="mismatch" 触发. face C++ 部署如果还在跑, 它继续用自己
的 env, 互不干扰.

**实测在 322 photo customer ground truth (1 真造假 + 39 clean)** (docs/cross_validation_v0_4_0.md):

| | face C++ baseline | face-py A.5 |
|---|---|---|
| Recall (catch 真造假) | 100% | **100%** |
| Specificity (clean session 不误报) | 92.3% | **94.9%** |
| Accuracy | 92.5% | **95.0%** |
| photo-level inconclusive rate | ~10% | 23% |

哲学: "图像质量不行就明确说不行" — face-py 主动用 quality gates 拒绝在不可信数据上猜
match/mismatch. 客户拿到的 anomaly 比 face C++ 更可信 (FP 少), 代价是 13 pp 更多 photo
走 inconclusive (但这是诚实, 比"错报代训"破坏性小一万倍).

### `POST /api/v1/face/session_check` (**Phase A.7 新, face-py 净增**)

两阶段 session-level identity verification. 替代 TA orchestrator 调 8 次 identity_check
+ 自己聚合的脆弱模式 — 把"内部一致性 + vs ref" 整套塞进一个 endpoint, 让 face-py
拥有 session-level decision authority.

请求:
```json
{ "ref_image_path": "/var/lib/.../ref.jpg",
  "photos": [
    {"sequence_no": 1, "photo_type": "sign_in",  "image_path": "/.../seq_001_sign_in.jpg"},
    {"sequence_no": 2, "photo_type": "process",  "image_path": "/.../seq_002_process.jpg"},
    ...
  ] }
```

响应:
```json
{ "session_status": "mismatch",
  "internal_consistency": "inconsistent",
  "outlier_sequence_nos": [11],
  "session_cos_to_ref": 0.664,
  "session_l2_to_ref": 0.820,
  "n_photos": 11, "n_post_gate": 9,
  "reason": "Stage 1: 1 outlier(s) mean_cos<0.2",
  "photo_results": [
    {"sequence_no": 1, "photo_type": "sign_in", "passes_gate": false,
     "error_code": "face_too_blurry", ...},
    {"sequence_no": 11, "photo_type": "sign_out", "passes_gate": true,
     "cosine_score": 0.141, "match_status": "mismatch",
     "is_outlier": true, "mean_cos_to_peers": 0.106, ...},
    ...
  ],
  "elapsed_ms": 374.4 }
```

**两阶段算法** (详 `module/face/pipeline.py::session_check` + `module/face/session_consistency.py`):
- **Stage 1 (internal_consistency)**: pairwise cos among post-gate photo embeddings.
  每张 photo 算它对其它 photo 的 mean cos; mean cos < `FACE_SESSION_OUTLIER_MEAN_COS`
  (默认 0.20) → 标 outlier. 任一 outlier → session_status='mismatch' (代训信号,
  不等 Stage 2).
- **Stage 2 (prototype vs ref)**: 仅在 Stage 1 通过时跑. 取 consistent core 的 mean
  embedding (L2 normalize) → cos/l2 vs ref → A.5 tri-state classify (cos<0.15 mismatch,
  cos≥0.30+l2≤1.15 match, else inconclusive).

兜底: 0 post-gate photo → inconclusive. 1 post-gate photo → skip Stage 1, Stage 2 直接跑.

**实测在 40-session customer GT 上 (Phase A.7 prototype)**:
| | TP | FN | FP | TN | cross-student FP rate |
|---|---|---|---|---|---|
| face C++ single-photo ANY rule | 1 | 0 | 3 | 36 | 4.80% |
| face-py A.5 single-photo cos≥0.30 | 1 | 0 | 2 | 37 | 1.58% |
| **face-py A.7 session_check** | **1** | **0** | **0** | **39** | **1.18%** |

rec#2 (S177509932310186, 唯一已知真造假) 的 imposter sign_out mean_cos=0.106, 其它
8 张真学员 mean_cos 0.685-0.747 — gap 0.5+, X=0.20 在安全 margin 中段.

### `POST /api/v1/face/compare`

跟 identity_check 类似但**没 head pose gate** (用户主动比对场景, 不滤). 字段 `is_same_person`
跟 face C++ 对齐.

### `GET /api/v1/health`

健康检查, 返 status / version / uptime. ENV `FACE_HTTP_AUTH_TOKEN` 不要求 (这个端点不验权).

## 模型

- **detect**: `models/face_detection_yunet_2023mar.onnx` (OpenCV Zoo). 1×3×640×640 输入,
  多尺度 anchor 输出 score/bbox/kps 各 3 个分辨率 (8/16/32 stride). 用最高 score 那张
  脸作为主脸.
- **recognize**: `models/face_recognition_sface_2021dec.onnx` (OpenCV Zoo). 1×3×112×112
  输入, 128-d 嵌入. 跟 face C++ 用的 `face-reidentification-retail-0095` (256-d) **不是同一模型** —
  cos 分布会变, threshold 0.4 要在 v0.4.0 验证阶段重测 + 重 tune.

## ENV — 命名跟 face / TA 对齐 (别 rename)

| ENV | 默认 | 说明 |
|---|---|---|
| `FACE_HTTP_HOST` | 127.0.0.1 | dev 不对外 |
| `FACE_HTTP_PORT` | 32192 | face C++ 占 32186, face-py 走 32192 |
| `FACE_HTTP_AUTH_TOKEN` | (empty) | 配了就启 auth, 跟 face C++ 同协议 (Bearer / X-API-Key) |
| `FACE_HTTP_AUTH_REQUIRED` | auto | 跟 `auth_token` 非空联动 |
| `FACE_COSINE_MISMATCH_THRESH` | **0.15** | cos < 此值 → mismatch (Phase A.5) |
| `FACE_COSINE_THRESH` | **0.30** | cos ≥ 此值 + l2 OK → match. 跟 face C++ 默认 0.4 不通用 |
| `FACE_L2_THRESH` | **由 cos 推导** | match 要求的 l2 上限. 🆕 不显式设置时 = `sqrt(2-2*FACE_COSINE_THRESH)` (issue #3). **l2 不是独立判据** — 归一化 embedding 下 `l2²=2-2cos`, 两个条件里只有更严的在生效. 旧默认 1.15 隐含 `cos>=0.3388`, 让 `FACE_COSINE_THRESH=0.30` 形同虚设. 显式设置仍生效, 但不自洽时启动 WARN |
| `FACE_DET_SCORE_MIN` | **0.88** | YuNet det_score < 此值 → inconclusive (Phase A.5) |
| `FACE_BBOX_MIN_PX` | **40** | bbox min(w,h) < 此值 → inconclusive (Phase A.5) |
| `FACE_CROP_CLARITY_MIN` | **30** | aligned crop Laplacian var < 此值 → inconclusive (Phase A.5) |
| `FACE_SESSION_OUTLIER_MEAN_COS` | **0.20** | session_check Stage 1: photo mean_cos<此值→outlier→代训 (Phase A.7) |
| `FACE_DETECT_MODEL_PATH` | models/face_detection_yunet_2023mar.onnx | YuNet |
| `FACE_RECOGNIZE_MODEL_PATH` | models/face_recognition_sface_2021dec.onnx | SFace |
| `FACE_DEVICE` | cuda | "cuda" / "cpu". cuda 走 CUDAExecutionProvider, 没卡时 fallback cpu |
| `FACE_LOG_LEVEL` | INFO | DEBUG / INFO / WARN |
| `FACE_LOG_REQUEST_BODIES` | 0 | 调试用. 默认 0 (隐私+日志体积). 跟 face C++ #15 patch 对齐. |

## 客户契约 — 不能乱改

照搬 face C++ 对外行为. 任何字段 / shape 变化都是 break:
- 成功路径填 cosine_score + l2_distance, **不**填 clarity_score
- error_code 路径填 error / error_code, **不**填 cosine_score / l2_distance
- error_code in {`no_face`, `pose_excessive`} **不**填 clarity_score (这俩走 explicit-return 分支)
- error_code in {`image_read_failed`, `ref_image_read_failed`, `feature_extraction_failed`} **填** clarity_score
- face_count **总是**填 (即便 no_face → 0; ref 读失败 → undefined 但 schema 里仍要 emit)

这些行为 1:1 抄 `/home/algo/face/service/face_http_server.cpp:process_training_photo` (line ~2580-2640) + `face-detection-0205` + recognizer 错误处理. 改任何一条都要先验证客户端 (TA `_populate_identity`) 不依赖.

## 决策语义 — quality gates 哲学

**核心 framing** (Phase A.5 引入): "图像质量不行就明确说不行".

face C++ 之前的行为依赖 face-detection-0205 偶然偏保守 (低质量图常返 no_face), 让"看
不清就不报警" 是 implicit 后果. face-py 选择 YuNet 检测更敏感, 检到的 face 更多, 这
本来是好事 — 但直接套用单 cos threshold 会让低质量 photo (模糊 / 太小 / 远 / 极端
pose) 上的不稳定 embedding 触发 mismatch, 错报代训 (Phase A.4 失败教训, specificity
跌到 54%).

A.5 redesign 走的路:
1. **显式 quality gates** (detector confidence / face size / aligned-crop clarity /
   pose) — 任一不过, 直接 inconclusive, 不参与 cos 决策
2. **cos 中间区**: 即便过了 quality gates, cos ∈ [0.15, 0.30) 也走 inconclusive
3. 只有 quality 好 + cos 极端 (< 0.15 OR ≥ 0.30) 才下 match/mismatch 结论

> 🆕 **l2 与 cos 不是两个判据** (issue #3): embedding 归一化后 `l2 = sqrt(2-2cos)`,
> `cos >= X and l2 <= Y` 里永远只有更严的那个生效。历史默认 (cos 0.30 + l2 1.15) 不自洽,
> 实际门槛一直是 `cos >= 0.3388` —— 现网被判 inconclusive 的照片 `cosine_score` 最大值
> 恰好 0.339 就是这个原因。现在 l2 默认由 cos 推导, 改 `FACE_COSINE_THRESH` 立即生效。

实测结果: recall 100% (catch 真造假), specificity 94.9% (vs face C++ 92.3%), 代价是
photo-level inconclusive 23% (vs face C++ ~10%). **多 13 pp inconc** 换 **少 1 个 false
positive 学员被错查**.

跟客户的话术: "AI 在能看清的 photo 上做了决定, 看不清的 photo 明确标 inconclusive 让
你决定要不要人工复核, 不会把好学员错报代训."

## v0.4.0 验证计划

跟 v0.3.0 client validation 同款流程:
1. face-py up 后, 跑 322 photo 跨验证:
   - face C++ 输出 → baseline.json (我们已有 `output/_actual/`)
   - face-py 输出 → new.json
   - per-photo diff: face_count / bbox IoU / cosine 分布
2. cosine threshold 重 tune — SFace 跟 OV reid 的 cos 分布大概率不一样,
   `FACE_COSINE_THRESH` 默认值要重新选 (跟 client 标注命中率重对)
3. v0.4.0 release notes: 把 cos 分布漂移、threshold 改动、命中率变化全列清楚
4. 客户验证窗口: ship 后跑客户机 docker 一周, 比 has_identity_anomaly 误报 / 漏报

不通过的话**不切默认** — TA 的 `FACE_HTTP_URL` 仍指 face C++ :32186 直到验证通过.

## 工程结构 (规划, Phase A 落)

```
service/
  serve.py                  HTTP server (stdlib ThreadingHTTPServer), 跟 cloth/day_night 同骨架
module/face/
  detector.py               YuNet ONNX 包装: load + preprocess + run + postprocess (NMS + best face)
  recognizer.py             SFace ONNX 包装: 112×112 → 128d embedding
  align.py                  5-point similarity transform 给定 landmarks + 目标尺寸 (Kazemi/InsightFace 同款)
  pose_gate.py              head pose 估计 + yaw/pitch gating (1:1 port from face C++)
  pipeline.py               identity_check / compare 顶层 compose
  errors.py                 error_code 枚举 + 跟 C++ 对齐的 message
tests/
  test_align.py
  test_pose_gate.py
  test_pipeline.py
  test_serve.py
models/                     ONNX 模型放这, gitignored, 由 setup script 拉
deploy/
  Dockerfile
  systemd/face-py.service
  env/dev.env.example
  env/prod.env.example
docs/
  cross_validation_v0_4_0.md   v0.3.0 baseline vs face-py 数据对比
```

## TODO (结构性 / 跨 session 约束, 不放可 fix bug)

- segment endpoint 暂留 face C++, Python 化是 Phase E+ 单独 issue
- YuNet 5 个 landmarks → 5-point align 跟 face C++ `landmarks-regression-retail-0009` 输出
  的 5 点**坐标顺序可能不同** — Phase A.2 实现时 print 对比, 错的话改顺序 (面坐标顺序约定:
  左眼 / 右眼 / 鼻 / 左嘴角 / 右嘴角)
- SFace 输入预处理: BGR vs RGB / 归一化 mean/std — 跟 OpenCV Zoo 文档对齐, 别凭印象写
- head pose gate 阈值 (yaw 0.35 / pitch 0.55) 是 face#13 patch 后的值, **不要**回到老 0.2/0.4
  (那是 v0.1.0 严口径, 已被废)

## 工作流约定 (来自 global CLAUDE.md, 这里强调)

- 发现 actionable bug → 立刻 `gh issue create --repo cvidkal/face-py`, 不写 TODO 注释
- 这个 CLAUDE.md 的 TODO 段**只放结构性约束** (上面 segment 留 C++、landmarks 坐标顺序、
  pose gate 阈值由来), 不放 actionable 问题
