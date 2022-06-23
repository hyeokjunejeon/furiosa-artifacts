import numpy as np
import pytest
import yaml
from helpers.util import InferenceTestSessionWrapper

import artifacts


def sanity_check_for_dvc_file(model, dvc_file_path: str):
    assert model
    assert yaml.safe_load(open(dvc_file_path).read())["outs"][0]["size"] == len(
        model.model
    )


@pytest.mark.asyncio
async def test_mlcommons_resnet50():
    sanity_check_for_dvc_file(
        await artifacts.MLCommonsResNet50(),
        "models/mlcommons_resnet50_v1.5_int8.onnx.dvc",
    )


@pytest.mark.asyncio
async def test_mlcommons_ssd_mobilenet():
    sanity_check_for_dvc_file(
        await artifacts.MLCommonsSSDMobileNet(),
        "models/mlcommons_ssd_mobilenet_v1_int8.onnx.dvc",
    )


@pytest.mark.asyncio
async def test_mlcommons_ssd_resnet34():
    sanity_check_for_dvc_file(
        await artifacts.MLCommonsSSDResNet34(),
        "models/mlcommons_ssd_resnet34_int8.onnx.dvc",
    )


@pytest.mark.asyncio
async def test_mlcommons_ssd_resnet34_perf():
    m: Model = await artifacts.MLCommonsSSDResNet34()
    test_image_path = "tests/assets/cat.jpg"

    assert len(m.classes) == 81, f"Classes is 81, but {len(m.classes)}"
    with InferenceTestSessionWrapper(m) as sess:
        result = sess.inference(
            test_image_path, post_config={"confidence_threshold": 0.3}
        )

        true_bbox = np.array(
            [
                [264.24792, 259.05603, 699.12964, 474.65332],
                [221.0502, 123.12275, 549.879, 543.1015],
            ],
            dtype=np.float32,
        )
        true_classid = np.array([16, 16], dtype=np.int32)
        true_confidence = np.array([0.37563688, 0.8747512], dtype=np.float32)
        assert len(result) == 3, "ssd_resnet34 output must have 3"
        assert (
            np.sum(np.abs(result[0] - true_bbox)) < 1e-3
        ), "please check detected bbox"
        assert (
            np.sum(np.abs(result[1] - true_classid)) < 1e-3
        ), "please check models' performance, Cat(16)"
        assert (
            np.sum(np.abs(result[2] - true_confidence)) < 1e-3
        ), "please check models' confidence values"


@pytest.mark.asyncio
async def test_efficientnetv2_s():
    assert await artifacts.EfficientNetV2_S()


@pytest.mark.asyncio
async def test_efficientnetv2_m():
    assert await artifacts.EfficientNetV2_M()
