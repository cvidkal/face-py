"""error_code 枚举 + message 措辞.

设计原则 (Phase A.5 引入): "图像质量不行就明确说不行". face-py 主动加 quality gates,
任何一项不过 → match_status="inconclusive" + 标 error_code 让 TA 端 / 客户端能 audit
是哪类 gate 拒绝的.

跟 face C++ 的对照:

| error_code | face C++ 触发 | face-py 触发 | 业务语义 |
|---|---|---|---|
| image_read_failed | cv::imread 空 | cv2.imread 返 None | IO 失败 |
| ref_image_read_failed | 同上 | 同 | IO 失败 |
| no_face | detect 返空 | YuNet 找不到任何 face | 检测失败 |
| pose_excessive | yaw/pitch 超 ABS 阈值 | 同, env FACE_POSE_ABS_YAW/PITCH | 极端侧脸 |
| feature_extraction_failed | 通用 catch-all | 同 | 异常兜底 |
| **detection_low_confidence** | (无) | det_score < FACE_DET_SCORE_MIN (默认 0.88) | YuNet 自己说不可信 (新, Phase A.5) |
| **face_too_small** | (无) | bbox min(w,h) < FACE_BBOX_MIN_PX (默认 40) | source face 太小, SFace input 不可靠 (新) |
| **face_too_blurry** | (无) | aligned face clarity < FACE_CROP_CLARITY_MIN (默认 30) | aligned crop 太模糊 (新) |
| **cos_inconclusive_zone** | (无) | cos 落 [cos_low, cos_match) 之间 | cos 在不确定区, 类似 face#13 patch (新) |

所有 inconclusive 类 error_code 都让 match_status="inconclusive". TA 的 _populate_identity
不区分 specific code (业务上等价), error_code 仅用于日志 + 调试. 加新 code 不破坏
TA 现有行为, 业务向前兼容.
"""
from __future__ import annotations


# 真失败类
IMAGE_READ_FAILED = "image_read_failed"
REF_IMAGE_READ_FAILED = "ref_image_read_failed"
NO_FACE = "no_face"
FEATURE_EXTRACTION_FAILED = "feature_extraction_failed"

# 质量 gate 类 (Phase A.5 新加, 全 → inconclusive)
POSE_EXCESSIVE = "pose_excessive"
DETECTION_LOW_CONFIDENCE = "detection_low_confidence"
FACE_TOO_SMALL = "face_too_small"
FACE_TOO_BLURRY = "face_too_blurry"
COS_INCONCLUSIVE_ZONE = "cos_inconclusive_zone"


# message 模板
def msg_image_read_failed(path: str) -> str:
    return f"Cannot read image: {path}"


def msg_ref_image_read_failed(path: str) -> str:
    return f"Cannot read reference image: {path}"


MSG_NO_FACE = "No face detected in image"


def msg_pose_excessive(yaw: float, pitch: float) -> str:
    return (f"head pose excessive (yaw_ratio={yaw}, pitch_ratio={pitch}); "
             "skipped face match")


def msg_detection_low_confidence(score: float, threshold: float) -> str:
    return (f"face detection confidence too low (score={score:.3f} < {threshold:.3f}); "
             "skipped to avoid unreliable embedding")


def msg_face_too_small(min_side: float, threshold: float) -> str:
    return (f"detected face too small (min(w,h)={min_side:.0f}px < {threshold:.0f}px); "
             "skipped to avoid unreliable embedding from upscaled crop")


def msg_face_too_blurry(clarity: float, threshold: float) -> str:
    return (f"aligned face crop too blurry (clarity={clarity:.1f} < {threshold:.1f}); "
             "skipped to avoid unreliable embedding")


def msg_cos_inconclusive_zone(cos: float, low: float, match: float) -> str:
    return (f"cos={cos:.3f} in uncertainty zone [{low:.2f}, {match:.2f}); "
             "neither confident match nor confident mismatch")
