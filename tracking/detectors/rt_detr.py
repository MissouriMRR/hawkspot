try:
    import numpy as np
    import onnxruntime
    import torch
    import torchvision
    from PIL.Image import Image
except ImportError:
    raise ImportError(
        "You are missing required dependencies for RTDETRV2_Detector. Required libraries: numpy, onnxruntime, torch, torchvision, PIL"
    )

import functools
import os
from typing import Self, Sequence, cast

from cv2.typing import MatLike

from tracking import COCO_CLASSES, BaseDetector, BoundingBox


class RTDETRV2_Detector(BaseDetector):
    @functools.total_ordering
    class ClassDetection(BoundingBox):
        class_name: str
        confidence: float

        def __init__(
            self,
            x: int,
            y: int,
            w: int,
            h: int,
            class_name: str,
            confidence: float,
        ):
            super().__init__(x, y, w, h)
            self.class_name = class_name
            self.confidence = confidence

        def __lt__(self, other: Self) -> bool:
            return self.confidence < other.confidence

        def __str__(self) -> str:
            return f"{self.class_name} ({self.confidence:.2f}) @ ({self.x}, {self.y}, {self.width}, {self.height})"

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.6,
        class_name: str | None = None,
        log_results: bool = False,
    ):
        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        print(model_path)

        self.model = onnxruntime.InferenceSession(model_path, opt_session, providers)

        self.model_output: Sequence[onnxruntime.NodeArg] = self.model.get_outputs()
        self.output_names: list[str] = [
            self.model_output[i].name for i in range(len(self.model_output))
        ]
        self.input_shape: list[int] = self.model.get_inputs()[0].shape
        self.input_height: int = self.input_shape[2]
        self.input_width: int = self.input_shape[3]

        self.confidence_threshold: float = confidence_threshold
        self.log_results: bool = log_results
        self.class_name: str | None = class_name

    def _convert_image(self, image: MatLike) -> torch.Tensor:
        """
        Convert an image read through cv2 to a resized tensor for model input.

        Parameters
        ----------
        image : MatLike
            The image to convert.

        Returns
        -------
        torch.Tensor
            The converted image, scaled from 0 to 1, and to the input size of the model.
        """

        image_pil: Image = torchvision.transforms.ToPILImage("RGB")(image)
        # Convert the cv2 image to RGB and resize it to the input size of the model
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.input_height, self.input_width)),
                torchvision.transforms.ToTensor(),
            ]
        )

        input_tensor: torch.Tensor = cast(torch.Tensor, transforms(image_pil))[None]
        return input_tensor

    def _parse_outputs(
        self,
        outputs: Sequence[np.ndarray],
        input_shape: tuple[int, int],
    ) -> Sequence[BoundingBox]:
        """
        Parse the outputs of the model to a bounding box.

        Parameters
        ----------
        outputs : list[np.ndarray]
            The outputs of the model.

        Returns
        -------
        BoundingBox
            The bounding box.
        """
        [labels], [boxes], [scores] = outputs
        detections: list["RTDETRV2_Detector.ClassDetection"] = []
        for i, box in enumerate(boxes):
            horizontal_scale = input_shape[1] / self.input_width
            vertical_scale = input_shape[0] / self.input_height

            # Scale box coordinates to fit input image size
            x = box[0] * horizontal_scale
            y = box[1] * vertical_scale
            width = (box[2] - box[0]) * horizontal_scale
            height = (box[3] - box[1]) * vertical_scale

            if (
                self.class_name is not None
                and COCO_CLASSES[labels[i]] != self.class_name
                or scores[i] < self.confidence_threshold
            ):
                # Not the class we're looking for
                continue
            detections.append(
                RTDETRV2_Detector.ClassDetection(
                    round(x),
                    round(y),
                    round(width),
                    round(height),
                    COCO_CLASSES[labels[i]],
                    scores[i],
                )
            )
        return detections

    def detect(self, image: MatLike) -> Sequence[BoundingBox]:
        input_tensor = self._convert_image(image)
        size_tensor = torch.tensor([self.input_width, self.input_height])[None]
        outputs = cast(
            Sequence[np.ndarray],
            self.model.run(
                self.output_names,
                input_feed={
                    self.model.get_inputs()[0].name: input_tensor.data.numpy(),
                    self.model.get_inputs()[1].name: size_tensor.data.numpy(),
                },
            ),
        )
        return self._parse_outputs(outputs, image.shape[:2])
