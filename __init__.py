import numpy as np
import torch.nn.functional as F
import torch
from comfy.utils import ProgressBar

from .dwpose import DWposeDetector


class DwposeTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_hands": ("BOOLEAN", {"default": True}),
                "show_body": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "tensorrt"

    def main(self, images, show_face, show_hands, show_body):

        pbar = ProgressBar(images.shape[0])
        dwpose = DWposeDetector()
        pose_frames = []

        for img in images:
            img_np_hwc = (img.cpu().numpy() * 255).astype(np.uint8)
            result = dwpose(image_np_hwc=img_np_hwc, show_face=show_face,
                            show_hands=show_hands, show_body=show_body)
            pose_frames.append(result)
            pbar.update(1)

        pose_frames_np = np.array(pose_frames).astype(np.float32) / 255
        return (torch.from_numpy(pose_frames_np),)


NODE_CLASS_MAPPINGS = {
    "DwposeTensorrt": DwposeTensorrt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DwposeTensorrt": "Dwpose Tensorrt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
