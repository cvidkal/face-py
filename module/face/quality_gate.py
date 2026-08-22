"""Quality gates — 不确定的数据明确走 inconclusive (Phase A.5 设计核心).

我们刻意构造的"诚实 inconclusive" 哲学: 与其在低质量数据上猜测一个 match/mismatch
(容易把好学员错报代训), 不如显式说 "我看不清". 让客户的人工复核流程聚焦在我们能
高置信判断的样本上.

实测在 322 photo customer ground truth 上:
- 不加 gate (单 cos threshold, face C++ 模式): specificity 69.2%, accuracy 70%
- 加 cos 中间区 (类似 face#13 patch): specificity 92.3%, accuracy 92.5% (= face C++)
- 加 det_score >= 0.88 + cos 中间区: **specificity 94.9%, accuracy 95% (好于 face C++)**
- 加 det >= 0.92 + cos 中间区: 100% specificity, accuracy 100% (但 inconclusive 60%)

最终选 (中) — recall 100% + specificity 94.9% + inconclusive 23% — pareto-optimal:
全方位优于 face C++ 但 inconclusive 率不夸张.

详见 docs/cross_validation_v0_4_0.md §"Phase A.5: quality gate retune".
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from .errors import COS_INCONCLUSIVE_ZONE

if TYPE_CHECKING:
    from .pipeline import SessionPhotoResult


CONSENSUS_COS_MIN = 0.30
PRODUCTION_COS_MATCH = 0.35
CONSENSUS_MIN_MATCH_PEERS = 2


def _env_num(name: str, default: float) -> float:
    """读数值型 env; **空串等同未设置**.

    docker compose 的 `${VAR:-}` 在 .env 里没这个键时传的是**空字符串**而不是
    不传, 而 os.environ.get(name, "10") 的默认值只在 key 缺失时生效 —— 于是
    float("") 直接 ValueError, 服务起不来。k8s 的可选 env 同理。
    (2026-08-06 dev 演练实测: cloth / face 双双 CrashLoop, 幸好没上 prod。)
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


# gate 的两种用法 (issue #1):
#   "block" — 历史行为: 任一 gate 不过 → 直接 inconclusive, 不做比对
#   "audit" — 默认: 照常比对、按 cos 定档, gate 结果作为**审计信号**透出去
#
# 改成 audit 的依据: 154 张人工确认「实际是本人」的照片 + 462 对**画质配平**的冒名对
# (同一张照片配别人的 ref) 实测, 四道 gate 对正负样本的拦截率**完全相同**
# (det 33.8%/33.8%, clarity 26.0%/26.0%, pose 25.3%/25.3%, bbox 10.4%/10.4%) ——
# 它们过滤的是"图好不好", 跟"是不是同一个人"正交, 所以只砍召回不买特异性:
#
#   现状 (block + cos>=0.30):  捞回 12.3%  假接受 1.30%
#   audit + cos>=0.35:        捞回 31.8%  假接受 0.22%   ← 两个维度同时更优
#
# 「诚实 inconclusive」的**意图**没变 — 不确定就别下结论。变的是用什么表达不确定:
# 不再用画质当代理, 而是用 cos 中间区。区间反而更宽了 (0.15~0.35 vs 0.15~0.30)。
GATE_MODE_BLOCK = "block"
GATE_MODE_AUDIT = "audit"


def session_match_consensus_enabled() -> bool:
    """Whether the production-shaped session consensus helper is enabled.

    Deliberately separate from QualityGateConfig so existing config serialization
    and API shapes stay unchanged while the rollout flag remains off by default.
    """
    raw = os.environ.get("FACE_SESSION_MATCH_CONSENSUS")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true"}


def _eligible_for_session_match_consensus(
        photo: "SessionPhotoResult",
        peers: Sequence["SessionPhotoResult"],
) -> bool:
    if photo.match_status != "inconclusive":
        return False
    if photo.error_code not in ("", COS_INCONCLUSIVE_ZONE):
        return False
    if photo.quality_flags:
        return False
    if photo.cosine_score is None:
        return False
    if not (CONSENSUS_COS_MIN <= photo.cosine_score < PRODUCTION_COS_MATCH):
        return False
    if photo.is_outlier:
        return False

    match_peer_count = 0
    mismatch_peer_count = 0
    outlier_peer_count = 0
    for peer in peers:
        if peer.is_outlier:
            outlier_peer_count += 1
        if peer.match_status == "match":
            match_peer_count += 1
        elif peer.match_status == "mismatch":
            mismatch_peer_count += 1

    return (
        match_peer_count >= CONSENSUS_MIN_MATCH_PEERS
        and mismatch_peer_count == 0
        and outlier_peer_count == 0
    )


def apply_session_match_consensus(photo_results: Sequence["SessionPhotoResult"]) -> int:
    """Promote eligible per-photo inconclusive results to match.

    This is a disabled-by-default production-shaped helper for later isolated
    evaluation. It only mutates `match_status` on eligible photos and returns the
    number of changed rows.
    """
    items = list(photo_results)
    changed = 0
    for index, photo in enumerate(items):
        peers = items[:index] + items[index + 1:]
        if not _eligible_for_session_match_consensus(photo, peers):
            continue
        photo.match_status = "match"
        changed += 1
    return changed


@dataclass
class QualityGateConfig:
    """4 个 quality gate 的阈值, 全 env 可调.

    阈值默认值来自 322 photo customer validation 数据扫参 (sweep_gates.py).
    """
    # YuNet 自己对检测的 confidence. < 0.88 一般是模糊脸 / 部分脸 / 反光等不稳定 case.
    det_score_min: float = 0.88
    # source bbox min(width, height) 像素值. < 40px 等于在 < 40×40 → 112×112 强 upscale,
    # SFace embedding 不稳. 40 是 customer 数据 p10 附近 (大多数 face 40-100px).
    face_bbox_min_px: float = 40.0
    # aligned 112×112 crop 的 Laplacian variance. < 30 是糊到看不清五官. 30 比较松,
    # customer 数据 p10 是 65 — 30 这个阈值仅 cut 真糊照, 不误伤一般 dashcam 质量.
    face_crop_clarity_min: float = 30.0
    # 挖掉 OSD 带后中心区的 Laplacian variance 下限 — 低于它认为"整张图没内容"
    # (全黑 / 过曝纯白 / 只拍到车窗). 10.0 是 2 万张真实归档照片扫参 + 逐张看图定的,
    # 命中率 0.075%; 13~25 那一带还能看到人脸, 不能判. 详见 content_gate.py docstring.
    # 设 0 关闭该 gate. **跟分辨率有关**, 换图源要重新标定.
    #
    # ⚠️ 这个 gate **不受 mode 影响, 永远拦截** —— 它跟上面四个不是一回事:
    # 上面四个问"这张脸看得清吗"(实测跟身份正交, 所以降级为审计), 这个问"这张图有
    # 内容吗"。全黑/过曝图**根本没有脸可比**, 放行下去只会得到一个假的 cos。
    photo_content_min: float = 10.0
    # 上面四个 quality gate 是拦下判定 ("block") 还是只做审计标记 ("audit").
    # 恢复历史行为: FACE_QUALITY_GATE_MODE=block + FACE_COSINE_THRESH=0.30
    mode: str = GATE_MODE_AUDIT
    # head pose 阈值, 跟 face C++ #13 patch 后默认一致. 这俩在 pose_gate.py 也读
    # FACE_POSE_ABS_YAW / FACE_POSE_ABS_PITCH env, 跟这俩字段联动.
    yaw_max: float = 0.35
    pitch_max: float = 0.55

    @classmethod
    def from_env(cls) -> "QualityGateConfig":
        return cls(
            det_score_min=_env_num("FACE_DET_SCORE_MIN", 0.88),
            face_bbox_min_px=_env_num("FACE_BBOX_MIN_PX", 40.0),
            face_crop_clarity_min=_env_num("FACE_CROP_CLARITY_MIN", 30.0),
            photo_content_min=_env_num("FACE_PHOTO_CONTENT_MIN", 10.0),
            mode=(os.environ.get("FACE_QUALITY_GATE_MODE", GATE_MODE_AUDIT).strip().lower()
                  or GATE_MODE_AUDIT),
            yaw_max=_env_num("FACE_POSE_ABS_YAW", 0.35),
            pitch_max=_env_num("FACE_POSE_ABS_PITCH", 0.55),
        )

    @property
    def blocks(self) -> bool:
        """gate 不过时是否直接终止比对 (block 模式)."""
        return self.mode == GATE_MODE_BLOCK
