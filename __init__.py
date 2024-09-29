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
    RETURN_NAMES = ("IMAGE", "cropped_faces", "cropped_faces_lmks", "FACE_BBOXES",)
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "FACE_BBOXES",)
    FUNCTION = "main"
    CATEGORY = "tensorrt"

    def main(self, images, show_face, show_hands, show_body):

        pbar = ProgressBar(images.shape[0])
        dwpose = DWposeDetector()
        pose_frames = []
        cropped_faces_frames = []
        cropped_faces_lmks_frames = []
        face_bboxes = []

        for img in images:
            img_np_hwc = (img.cpu().numpy() * 255).astype(np.uint8)
            pose, face_bbox, cropped_faces, cropped_face_lmks = dwpose(image_np_hwc=img_np_hwc, show_face=show_face,
                            show_hands=show_hands, show_body=show_body)
            pose_frames.append(pose)

            if face_bbox:
                face_bboxes.append(face_bbox)

            # flatten list
            if cropped_faces:
                for cropped_face in cropped_faces:
                    cropped_faces_frames.append(cropped_face)
                for cropped_face_lmk in cropped_face_lmks:
                    cropped_faces_lmks_frames.append(cropped_face_lmk)

            pbar.update(1)

        pose_frames_np = np.array(pose_frames).astype(np.float32) / 255
        if cropped_faces_frames:
            cropped_faces_np = np.array(cropped_faces_frames, dtype=np.float32) / 255
            cropped_faces_lmks_frames_np = np.array(cropped_faces_lmks_frames, dtype=np.float32) / 255
        else:
            # black image
            cropped_faces_np = np.zeros((1, images.shape[1], images.shape[2], 3), dtype=np.float32) / 255
            cropped_faces_lmks_frames_np = np.zeros((1, images.shape[1], images.shape[2], 3), dtype=np.float32) / 255

        return (torch.from_numpy(pose_frames_np), torch.from_numpy(cropped_faces_np), torch.from_numpy(cropped_faces_lmks_frames_np), face_bboxes,)


NODE_CLASS_MAPPINGS = {
    "DwposeTensorrt": DwposeTensorrt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DwposeTensorrt": "Dwpose Tensorrt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
