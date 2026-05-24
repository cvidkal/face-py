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

响应 (per-photo 错误路径, 跟 face C++ 同款 error_code 枚举):
- `no_face` — face_count = 0, 没检到脸 (此分支 cosine/l2 不填; clarity 不填)
- `pose_excessive` — yaw > 0.35 OR pitch > 0.55 (跟 face C++ #13 patch 后阈值对齐;
  此分支 cosine/l2 不填; clarity 不填 — face C++ line ~2616 行为)
- `image_read_failed` / `ref_image_read_failed` — opencv 读图失败 (此分支带 clarity_score)
- `feature_extraction_failed` — embedding 抽取异常 (此分支带 clarity_score)

`match_status` ∈ `{match, mismatch, inconclusive}`. 阈值 `FACE_COSINE_THRESH` (默认
**0.30**, Phase A.4 重 tune 后, 见 docs/cross_validation_v0_4_0.md) + `FACE_L2_THRESH`
(默认 **1.10**) — 注意跟 face C++ 默认 0.4/1.0 **不一样**, 因 SFace 跟
face-reidentification-retail-0095 cos 分布偏移 ~ -0.10.

切到 face-py 时 TA 端 `FACE_COSINE_THRESH` env **必须同步切**, 否则带 face C++ 的
0.4 阈值打 face-py 会得到 55% 决策一致率 (Phase A.4 实测).

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
| `FACE_COSINE_THRESH` | **0.30** | match 阈值. **跟 face C++ 默认 0.4 不一样** (Phase A.4 重 tune) |
| `FACE_L2_THRESH` | **1.10** | match l2 阈值. 同上 |
| `FACE_COSINE_INCONCLUSIVE_THRESH` | 0.20 | inconclusive 中间区下界, face#13 中间区相对 cos 阈值 - 0.1 |
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
