"""error_code 枚举 + message 措辞跟 face C++ 对齐.

face C++ classify_image_validation_failure (line ~360 in face_http_server.cpp) 把
runtime exception 分级成几个稳定 error_code, 让 TA 客户端能枚举处理. 重写时
**字面 1:1** 保留, TA 端 _populate_identity 按 error_code 分支判断.

跟 face C++ 的对照表:

| error_code | face C++ 触发条件 | face-py 触发条件 |
|---|---|---|
| image_read_failed | cv::imread(image_path) 空 | cv2.imread / cv2.imdecode 返 None |
| ref_image_read_failed | ref image 空 (在 identity_check) | 同 |
| no_face | detect 返空 | 同 |
| pose_excessive | yaw 或 pitch 超过 ABS 阈值 | 同, 阈值 FACE_POSE_ABS_YAW/PITCH |
| feature_extraction_failed | 通用 catch-all (extract 抛异常) | 同 |
| image_quality_insufficient | detect 失败 + clarity < threshold | (暂不实现, 见下) |
| face_detection_failed | detect 失败 + clarity >= threshold | (同上, 简化为 no_face) |

**简化说明**: image_quality_insufficient / face_detection_failed 这俩是 face C++
里 detect 失败时根据 clarity 二分的细化 code. v0.4.0 face-py 先简化成 no_face,
clarity_score 字段仍写, TA 端 _populate_identity 现在的处理是把 no_face 和
其它 detect 失败都映射成 match_status="inconclusive", 业务上等价 — face C++ 那边
TA 也没区别对待这俩 code. Phase A.4 跨验证时若发现 client 有依赖这两 code 再补.
"""
from __future__ import annotations


# error_code 枚举
IMAGE_READ_FAILED = "image_read_failed"
REF_IMAGE_READ_FAILED = "ref_image_read_failed"
NO_FACE = "no_face"
POSE_EXCESSIVE = "pose_excessive"
FEATURE_EXTRACTION_FAILED = "feature_extraction_failed"


# message 模板, 跟 face C++ 字面对齐
def msg_image_read_failed(path: str) -> str:
    return f"Cannot read image: {path}"


def msg_ref_image_read_failed(path: str) -> str:
    return f"Cannot read reference image: {path}"


MSG_NO_FACE = "No face detected in image"


def msg_pose_excessive(yaw: float, pitch: float) -> str:
    # 跟 face C++ line 4374-4377 同款 ostringstream 输出
    return (f"head pose excessive (yaw_ratio={yaw}, pitch_ratio={pitch}); "
             "skipped face match")
