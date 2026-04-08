"""
Various utility functions that are useful internally
"""

import cv2

from tracking import BoundingBox


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
