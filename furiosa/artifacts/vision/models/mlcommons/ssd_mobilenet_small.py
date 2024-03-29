import logging
from typing import Any, Dict, ForwardRef, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import Field

from furiosa.registry import Model

from . import anchor_generator  # type: ignore[import]
from .common.datasets import coco

tensorArray = ForwardRef("tensor.TensorArray")

# https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L155-L158
PRIORS = np.concatenate(
    [
        tensor.numpy()
        # pylint: disable=protected-access
        for tensor in anchor_generator.create_ssd_anchors()._generate(
            [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]
        )
    ],
    axis=0,
)
PRIORS_Y1 = np.expand_dims(np.expand_dims(PRIORS[:, 0], axis=1), axis=0)
PRIORS_X1 = np.expand_dims(np.expand_dims(PRIORS[:, 1], axis=1), axis=0)
PRIORS_Y2 = np.expand_dims(np.expand_dims(PRIORS[:, 2], axis=1), axis=0)
PRIORS_X2 = np.expand_dims(np.expand_dims(PRIORS[:, 3], axis=1), axis=0)
PRIORS_WIDTHS = PRIORS_X2 - PRIORS_X1
PRIORS_HEIGHTS = PRIORS_Y2 - PRIORS_Y1
PRIORS_CENTER_X = PRIORS_X1 + 0.5 * PRIORS_WIDTHS
PRIORS_CENTER_Y = PRIORS_Y1 + 0.5 * PRIORS_HEIGHTS

del PRIORS_Y1, PRIORS_X1, PRIORS_Y2, PRIORS_X2, PRIORS


class SSDSmallConstant(object):
    PRIORS_WIDTHS = PRIORS_WIDTHS
    PRIORS_HEIGHTS = PRIORS_HEIGHTS
    PRIORS_CENTER_X = PRIORS_CENTER_X
    PRIORS_CENTER_Y = PRIORS_CENTER_Y


class MLCommonsSSDSmallModel(Model):
    """MLCommons MobileNet v1 model"""

    # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/tools/submission/submission-checker.py#L467

    @property
    def classes(self):
        return coco.MobileNetSSD_CLASSES

    def preprocess(self, image_path: str) -> Tuple[npt.ArrayLike, Dict[str, Any]]:
        """Read and preprocess an image located at image_path."""
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/main.py#L49-L51
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/dataset.py#L242-L249
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        image = np.array(image, dtype=np.float32)
        if len(image.shape) < 3 or image.shape[2] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width = image.shape[1]
        height = image.shape[0]
        image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_LINEAR)
        image -= 127.5
        image /= 127.5
        image = image.transpose([2, 0, 1])
        return image[np.newaxis, ...], {"width": width, "height": height}

    def sigmoid(self, x: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
        return 1 / (1 + np.exp(-x))

    def decode_boxes(self, rel_codes: np.ndarray) -> np.ndarray:
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L149-L198

        # pylint: disable=invalid-name

        dy = np.expand_dims(rel_codes[:, :, 0], axis=2)
        dx = np.expand_dims(rel_codes[:, :, 1], axis=2)
        dh = np.expand_dims(rel_codes[:, :, 2], axis=2)
        dw = np.expand_dims(rel_codes[:, :, 3], axis=2)

        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L127
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L166
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L177-L180
        dx /= 10.0
        dy /= 10.0
        dw /= 5.0
        dh /= 5.0

        prediction_center_x = dx * SSDSmallConstant.PRIORS_WIDTHS + SSDSmallConstant.PRIORS_CENTER_X
        prediction_center_y = (
            dy * SSDSmallConstant.PRIORS_HEIGHTS + SSDSmallConstant.PRIORS_CENTER_Y
        )
        prediction_w = np.exp(dw) * SSDSmallConstant.PRIORS_WIDTHS
        prediction_h = np.exp(dh) * SSDSmallConstant.PRIORS_HEIGHTS

        prediction_boxes = np.concatenate(
            (
                prediction_center_x - 0.5 * prediction_w,
                prediction_center_y - 0.5 * prediction_h,
                prediction_center_x + 0.5 * prediction_w,
                prediction_center_y + 0.5 * prediction_h,
            ),
            axis=2,
        )
        return prediction_boxes

    def filter_results(
        self, scores: np.ndarray, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L197-L212
        selected_box_probs = []
        labels = []
        for class_index in range(1, scores.shape[1]):
            probs = scores[:, class_index]
            mask = probs > 0.3
            probs = probs[mask]
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate((subset_boxes, probs.reshape(-1, 1)), axis=1)
            box_probs = self.nms(box_probs, 0.6)
            selected_box_probs.append(box_probs)
            labels.append(np.full((box_probs.shape[0],), class_index, dtype=np.int64))
        selected_box_probs = np.concatenate(selected_box_probs)  # type: ignore[assignment]
        labels = np.concatenate(labels)  # type: ignore[assignment]
        return selected_box_probs[:, :4], labels, selected_box_probs[:, 4]  # type: ignore[call-overload, return-value]

    def nms(self, box_scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L122-L146
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)[::-1]
        while len(indexes) > 0:
            current = indexes[0]
            picked.append(current.item())
            if len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[1:]
            rest_boxes = boxes[indexes, :]
            iou = self.box_iou(rest_boxes, np.expand_dims(current_box, axis=0))
            indexes = indexes[iou <= iou_threshold]
        return box_scores[picked, :]

    def box_iou(self, boxes1: np.ndarray, boxes2: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Return intersection-over-union (Jaccard index) of boxes."""
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L103-L119
        overlap_left_top = np.maximum(boxes1[..., :2], boxes2[..., :2])
        overlap_right_bottom = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        overlap_area = self.box_area(overlap_left_top, overlap_right_bottom)
        area1 = self.box_area(boxes1[..., :2], boxes1[..., 2:])
        area2 = self.box_area(boxes2[..., :2], boxes2[..., 2:])
        return overlap_area / (area1 + area2 - overlap_area + eps)

    def box_area(self, left_top: np.ndarray, right_bottom: np.ndarray):
        """Compute the areas of rectangles given two corners."""
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/utils.py#L89-L100
        width_height = np.clip(right_bottom - left_top, a_min=0.0, a_max=None)
        return width_height[..., 0] * width_height[..., 1]

    def calibration_box(self, bbox, width, height):
        bbox[:, 0] *= width
        bbox[:, 1] *= height
        bbox[:, 2] *= width
        bbox[:, 3] *= height

        bbox[:, 2] -= bbox[:, 0]
        bbox[:, 3] -= bbox[:, 1]
        return bbox

    def postprocess(
        self, outputs: tensorArray, extra_params: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        logging.debug(f"extra_params: {extra_params}")
        outputs = [output.numpy() for output in outputs]
        assert len(outputs) == 12, len(outputs)
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L94-L97
        class_logits = [
            output.transpose((0, 2, 3, 1)).reshape((1, -1, 91)) for output in outputs[0::2]
        ]
        box_regression = [
            output.transpose((0, 2, 3, 1)).reshape((1, -1, 4)) for output in outputs[1::2]
        ]
        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L144-L166
        class_logits = np.concatenate(class_logits, axis=1)  # type: ignore[assignment]
        box_regression = np.concatenate(box_regression, axis=1)  # type: ignore[assignment]
        batch_scores = self.sigmoid(class_logits)  # type: ignore[arg-type]
        batch_boxes = self.decode_boxes(box_regression)  # type: ignore[arg-type]

        # https://github.com/mlcommons/inference/blob/de6497f9d64b85668f2ab9c26c9e3889a7be257b/vision/classification_and_detection/python/models/ssd_mobilenet_v1.py#L178-L185
        filtered_boxes, filtered_labels, filtered_scores = [], [], []
        width = extra_params["width"]
        height = extra_params["height"]
        for scores, boxes in zip(batch_scores, batch_boxes):
            boxes, labels, scores = self.filter_results(scores, boxes)
            filtered_boxes.append(self.calibration_box(boxes, width, height))
            filtered_labels.append(labels)
            filtered_scores.append(scores)
        return filtered_boxes[0], filtered_labels[0], filtered_scores[0]  # 1-batch(NPU)


MLCommonsSSDSmallModel.update_forward_refs()
