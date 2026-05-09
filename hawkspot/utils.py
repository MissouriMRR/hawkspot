"""
Various utility functions that are useful internally
"""

import cv2

from hawkspot import BoundingBox, DetectionOffset


def draw_bounding_box(
    image: cv2.typing.MatLike,
    bounding_box: BoundingBox,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """
    Draws a bounding box at a given location on an image.
    """
    p1, p2 = bounding_box.to_pts()
    cv2.rectangle(
        image,
        p1,
        p2,
        color,
        2,
    )


def draw_offset_line(image: cv2.typing.MatLike, offset: DetectionOffset) -> None:
    """
    Draws a line from the center of the image to the offset point.
    """
    center = (image.shape[1] // 2, image.shape[0] // 2)
    offset_pt = (center[0] + int(offset.x), center[1] + int(offset.y))
    cv2.line(image, center, offset_pt, (0, 255, 0), 2)
