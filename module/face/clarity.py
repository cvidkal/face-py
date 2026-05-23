"""Image clarity score = Laplacian variance.

1:1 port from face C++ compute_image_clarity_score (service/face_http_server.cpp:340).
TA `identity_validation.clarity_score` 在 error_code 路径 (image_read_failed /
feature_extraction_failed) 才暴露; success / no_face / pose_excessive 路径不填.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def compute_image_clarity_score(image: Optional[np.ndarray]) -> Optional[float]:
    """返回 Laplacian variance, image 空时返 None.

    跟 face C++:
      cv::Laplacian(gray, laplacian, CV_64F);
      cv::meanStdDev(laplacian, mean, stddev);
      return stddev[0] * stddev[0];

    数值上等价 numpy 的 laplacian.var() (= mean of (x - mean)^2). cv::meanStdDev
    的 stddev 是 population stddev (divisor=N), 跟 numpy.var(ddof=0) 一致.
    """
    if image is None or image.size == 0:
        return None
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # ddof=0 跟 cv::meanStdDev 一致 (population variance, 不是 sample variance)
    return float(np.var(laplacian))
