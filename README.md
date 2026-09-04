# face-py

Face inference HTTP service — Python + ONNX Runtime + CUDA 12.

替代 [`cvidkal/jiapei-anticheat`](https://github.com/cvidkal/jiapei-anticheat) 的 C++
`face_http_server` (除 `/face/segment` 外), 接入 [`cvidkal/training_analyzer`](https://github.com/cvidkal/training_analyzer) 6-服务部署栈.

跟 cloth / day_night / dup_detect / parked_fraud 一致的 Python 服务形态.

## endpoints

| Method | Path | 用途 |
|---|---|---|
| POST | `/api/v1/face/identity_check` | 单张图 face detect + recognize + 跟 ref 比. TA `/training/analyze` 主流程逐张调. |
| POST | `/api/v1/face/compare` | 两张图直接比 cosine. TA `/compare/student` + `/compare/tasks` 用. |
| GET | `/api/v1/health` | 健康检查 (status / version / uptime). |

`/face/segment` **不在本仓** — 留在 face C++ 直到单独 Python 化, 详见 CLAUDE.md.

## 快速开始

```bash
# 1. 装依赖 (推荐 venv)
pip install -r requirements.txt

# 2. 模型 (gitignored, 用前需要拉一次)
mkdir -p models
# 暂时手动从 jiapei-anticheat/models/ 拷贝, 后续会有 download script:
cp /home/algo/face/models/face_detection_yunet_2023mar.onnx models/
cp /home/algo/face/models/face_recognition_sface_2021dec.onnx models/

# 3. 启动 dev mode
./run_http_server.sh --env dev

# 4. 健康检查
curl http://127.0.0.1:32192/api/v1/health
```

## perf 目标

| 指标 | face C++ (OV CPU) | face-py (ORT CUDA) |
|---|---|---|
| `/identity_check` per call | ~40 ms | ~10 ms (估, Phase A 量) |
| 8 photo / session 累计 | ~320 ms | ~80 ms |
| TA e2e (URL 入参) | 1.1 s | **~0.85 s 估** (砍 240 ms) |

实际数字 Phase A.4 跑 322 photo bench 后填.

## 客户契约

跟 face C++ byte-level 兼容. 字段 / shape / error_code / behavior 全保留. 切换时
TA 的 `FACE_HTTP_URL` 改成 face-py:32192, 客户端 / 监管平台**零感知**.

详见 [CLAUDE.md](./CLAUDE.md).

## 进度 / 上下文

- 上游 tracking: cvidkal/training_analyzer#41
- pivot rationale: cvidkal/jiapei-anticheat#20 (closed superseded)
- perf umbrella: cvidkal/training_analyzer#39

Phase A.1 (本 commit): 骨架. 服务可启, /health 通, identity_check / compare 返 501 stub.
Phase A.2-3: 真实现 identity_check + compare.
Phase A.4: 跨验证 vs face C++ baseline.
Phase A.7+: `FACE_SESSION_MATCH_CONSENSUS=1` 可开启一个**默认关闭**的 production-shaped
per-photo 共识候选: 仅在 multi-photo session 的 Stage 1 判定为 `consistent` 后, 把满足
`[0.30, 0.35)` / 无 quality flag / 无 outlier peer / ≥2 clean match peers 的单张
`inconclusive` 提升为 `match`. 默认配置保持关闭, 不影响现网客户契约.
Phase B+: docker, deploy bundle, 切默认.
