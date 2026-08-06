"""Content gate — 判「这张照片根本没拍到内容」, 区别于「拍到了但看不清」.

## 为什么单独一档

quality gate (quality_gate.py) 回答的是「这两张脸像不像我判不了」, 全部收敛成
match_status="inconclusive"。但客户实际遇到的是**另一件事**: 抓拍设备偶尔产出
全黑 / 过曝纯白 / 只拍到车窗的照片 —— **整张图没有任何内容**, 谈不上"看不清人脸",
是"压根没有人脸可看"。

两者混在 inconclusive 里, 审核页就没法区分, 审核员只能在「匹配/不匹配/待定」里
硬挑一个。TA 现网 359 条人工回标里 157 条是 inconclusive → 被人工改成 match,
其中相当一部分就是这个原因 (见 issue #1)。

## 判据: 挖掉 OSD 后中心区的 Laplacian variance

抓拍图带**烧录的 OSD 红字** (时间/机构/学员 在上, 经纬度/车牌/车速 在下)。
全黑图也有亮红字, 直接算全图 Laplacian / 对比度会被红字撑高, 看不出"没内容"。
所以:

1. 裁掉上下各 `_OSD_BAND_RATIO` 的 OSD 带
2. 中心区里残留的强红像素抹成中值 (红字可能溢出到中间)
3. 对剩下的灰度图算 Laplacian variance = `content_score`

**「暗」不是判据** —— 夜间训练照本来就暗且完全合法 (30 万张归档实测: 全图
dark_frac ≥ 0.8 的占 1.27%, 逐张看过去绝大多数人脸清清楚楚)。用亮度判会把合法
夜训照大批误杀。而 Laplacian 只看**有没有结构**, 对全黑和过曝纯白两种失效都低,
一条规则覆盖两种。

## 阈值标定 (2 万张真实归档照片, 320×240)

| content_score < | 命中率 |
|---|---|
| 8  | 0.035% |
| 10 | 0.075% |
| 12 | 0.100% |
| 20 | 0.225% |

逐张看图: **< 10 基本是纯粹的「除 OSD 红字外什么都没有」**; 13~25 那一带已经能
看到人脸 (只是暗), **不能判 unusable**。默认取 10.0 —— 刻意保守, 宁可漏也不误杀:
误判一张合法夜训照为"异常照片", 代价是客户学时被无故质疑。

**Laplacian variance 跟分辨率有关** — 阈值是在 320×240 抓拍图上标的。如果以后
接入别的分辨率的图源, 要重新扫参 (env `FACE_PHOTO_CONTENT_MIN` 可调, 设 0 关闭)。
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# OSD 红字带占图像高度的比例 (上下各一条). 320×240 的抓拍图上下各约 2 行字。
_OSD_BAND_RATIO = 0.16

# 强红像素判据 — 烧录 OSD 是纯红色, 跟车内暖色调 / 尾灯区分开需要"红得压过其它通道"
_RED_MIN = 90
_RED_DOMINANCE = 45


def compute_content_score(image: Optional[np.ndarray]) -> Optional[float]:
    """挖掉 OSD 带 + 抹掉残留红字后, 中心区的 Laplacian variance.

    image 为空 / 尺寸小到裁不出中心区时返 None (调用方按"判不了"处理, 不判 unusable)。
    """
    if image is None or getattr(image, "size", 0) == 0:
        return None
    h = image.shape[0]
    top, bottom = int(h * _OSD_BAND_RATIO), int(h * (1.0 - _OSD_BAND_RATIO))
    if bottom - top < 8:      # 太小的图裁完没意义, 不判
        return None
    core = image[top:bottom]

    if core.ndim == 2:
        gray = core
    else:
        gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
        b, g, r = (core[:, :, 0].astype(np.int16),
                   core[:, :, 1].astype(np.int16),
                   core[:, :, 2].astype(np.int16))
        red = (r > _RED_MIN) & (r - g > _RED_DOMINANCE) & (r - b > _RED_DOMINANCE)
        if red.any():
            rest = gray[~red]
            # 整块都是红字 (理论上不该发生) → 没有可参考的背景, 直接判没内容
            gray = gray.copy()
            gray[red] = int(np.median(rest)) if rest.size else 0

    return float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))


def is_photo_unusable(content_score: Optional[float], threshold: float) -> bool:
    """content_score 低于阈值 = 整张图没内容. threshold <= 0 表示关闭该 gate.

    score 为 None (算不出) **不算** unusable — 宁可放行给后面的 detect 去判,
    也不要因为一个辅助指标算不出来就否掉整张照片。
    """
    if threshold <= 0 or content_score is None:
        return False
    return content_score < threshold
