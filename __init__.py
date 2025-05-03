import numpy as np
import torch.nn.functional as F
import torch
import os
import folder_paths
from .dwpose import DWposeDetector

from comfy.utils import ProgressBar
from .trt_utilities import Engine
from .utilities import download_file, ColoredLogger
import comfy.model_management as mm
import time
import tensorrt

logger = ColoredLogger("ComfyUI-Dwpose-Tensorrt")


class DwposeTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "yolox_trt_model": ("YOLOX_TRT_MODEL",),
                "dwpose_trt_model": ("DWPOSE_TRT_MODEL",),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_hands": ("BOOLEAN", {"default": True}),
                "show_body": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "tensorrt"

    def main(self, images, yolox_trt_model, dwpose_trt_model, show_face, show_hands, show_body):
        yolox_trt_model.activate()
        yolox_trt_model.allocate_buffers()
        dwpose_trt_model.activate()
        dwpose_trt_model.allocate_buffers()

        pbar = ProgressBar(images.shape[0])
        dwpose = DWposeDetector(yolox_trt_model, dwpose_trt_model)
        pose_frames = []

        for img in images:
            img_np_hwc = (img.cpu().numpy() * 255).astype(np.uint8)
            result = dwpose(image_np_hwc=img_np_hwc, show_face=show_face,
                            show_hands=show_hands, show_body=show_body)
            pose_frames.append(result)
            pbar.update(1)

        yolox_trt_model.reset() # frees engine vram
        dwpose_trt_model.reset() # frees engine vram
        mm.soft_empty_cache()

        pose_frames_np = np.array(pose_frames).astype(np.float32) / 255
        return (torch.from_numpy(pose_frames_np),)


class LoadDwposeTensorrtModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "yolox_precision": (["fp16", "fp32"], {"default": "fp16", "tooltip": "Precision to build the yolox tensorrt engine"}),
                "dwpose_precision": (["fp16", "fp32"], {"default": "fp32", "tooltip": "Precision to build the dwpose tensorrt engine, FP32 is more accurate"}),
            }
        }
    RETURN_NAMES = ("yolox_trt_model","dwpose_trt_model",)
    RETURN_TYPES = ("YOLOX_TRT_MODEL","DWPOSE_TRT_MODEL")
    FUNCTION = "main"
    CATEGORY = "tensorrt"
    DESCRIPTION = "Load tensorrt models, they will be built automatically if not found."
    FUNCTION = "load_dwpose_tensorrt_model"
    
    def load_dwpose_tensorrt_model(self, yolox_precision, dwpose_precision):
        tensorrt_models_dir = os.path.join(folder_paths.models_dir, "tensorrt", "dwpose")
        onnx_models_dir = os.path.join(folder_paths.models_dir, "onnx", "dwpose")

        os.makedirs(tensorrt_models_dir, exist_ok=True)
        os.makedirs(onnx_models_dir, exist_ok=True)

        yolox_onnx_model_path = os.path.join(onnx_models_dir, "yolox_l.onnx")
        dwpose_onnx_model_path = os.path.join(onnx_models_dir, "dw-ll_ucoco_384.onnx")
        
        yolox_tensorrt_model_path = os.path.join(tensorrt_models_dir, f"yolox_l_{yolox_precision}_{tensorrt.__version__}.trt")
        dwpose_tensorrt_model_path = os.path.join(tensorrt_models_dir, f"dw-ll_ucoco_384_{dwpose_precision}_{tensorrt.__version__}.trt")

        # Download onnx & build tensorrt engines
        if not os.path.exists(yolox_tensorrt_model_path):
            if not os.path.exists(yolox_onnx_model_path):
                yolox_onnx_model_download_url = f"https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
                logger.info(f"Downloading {yolox_onnx_model_download_url}")
                download_file(url=yolox_onnx_model_download_url, save_path=yolox_onnx_model_path)
            else:
                logger.info(f"Yolox_l onnx model found at: {yolox_onnx_model_path}")

            # Build tensorrt engine
            logger.info(f"Building TensorRT engine for {yolox_onnx_model_path}: {yolox_tensorrt_model_path}")
            mm.soft_empty_cache()
            s = time.time()
            engine = Engine(yolox_tensorrt_model_path)
            engine.build(
                onnx_path=yolox_onnx_model_path,
                fp16= True if yolox_precision == "fp16" else False
            )
            e = time.time()
            logger.info(f"Time taken to build Yolox_L engine: {(e-s)} seconds")

        # Download onnx & build tensorrt engines
        if not os.path.exists(dwpose_tensorrt_model_path):
            if not os.path.exists(dwpose_onnx_model_path):
                dwpose_onnx_model_download_url = f"https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
                logger.info(f"Downloading {dwpose_onnx_model_download_url}")
                download_file(url=dwpose_onnx_model_download_url, save_path=dwpose_onnx_model_path)
            else:
                logger.info(f"Dwpose onnx model found at: {dwpose_onnx_model_path}")

            # Build tensorrt engine
            logger.info(f"Building TensorRT engine for {dwpose_onnx_model_path}: {dwpose_tensorrt_model_path}")
            mm.soft_empty_cache()
            s = time.time()
            engine = Engine(dwpose_tensorrt_model_path)
            engine.build(
                onnx_path=dwpose_onnx_model_path,
                fp16= True if dwpose_precision == "fp16" else False
            )
            e = time.time()
            logger.info(f"Time taken to build Dwpose engine: {(e-s)} seconds")

        # Load tensorrt model
        logger.info(f"Loading Yolox_L TensorRT engine: {yolox_tensorrt_model_path}")
        logger.info(f"Loading Dwpose TensorRT engine: {dwpose_tensorrt_model_path}")

        mm.soft_empty_cache()
        yolox_engine = Engine(yolox_tensorrt_model_path)
        dwpose_engine = Engine(dwpose_tensorrt_model_path)

        yolox_engine.load()
        dwpose_engine.load()

        return (yolox_engine, dwpose_engine,)

NODE_CLASS_MAPPINGS = {
    "DwposeTensorrt": DwposeTensorrt,
    "LoadDwposeTensorrtModels": LoadDwposeTensorrtModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DwposeTensorrt": "Dwpose Tensorrt ⚡",
    "LoadDwposeTensorrtModels": "Load Dwpose Tensorrt Models",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
