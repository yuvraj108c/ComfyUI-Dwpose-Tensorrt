import numpy as np
import torch.nn.functional as F
import torch
from comfy.utils import ProgressBar

from .dwpose import DWposeDetector
from .trt_utilities import Engine
import folder_paths
import os

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
        # setup tensorrt engines
        if (not hasattr(self, 'engine') or self.engine_label != "engine"):
            self.engine = Engine(os.path.join(folder_paths.models_dir, "tensorrt", "dwpose", "yolox_l.engine"))
            self.engine.load()
            self.engine.activate()
            self.engine.allocate_buffers()
            self.engine_label = "engine"

        if (not hasattr(self, 'engine2') or self.engine_label_2 != "engine2"):
            self.engine2 = Engine(os.path.join(
                folder_paths.models_dir, "tensorrt", "dwpose", "dw-ll_ucoco_384.engine"))
            self.engine2.load()
            self.engine2.activate()
            self.engine2.allocate_buffers()
            self.engine_label_2 = "engine2"

        image_count = images.shape[0] if torch.is_tensor(images) else len(images)
        pbar = ProgressBar(image_count)
        dwpose = DWposeDetector(self.engine, self.engine2)
        pose_frames = []
        cropped_faces_frames = []
        cropped_faces_lmks_frames = []
        face_bboxes = []

        for img in images:
            img_np_hwc = (img.cpu().numpy() * 255).astype(np.uint8)
            pose, face_bbox, cropped_faces, cropped_face_lmks = dwpose(image_np_hwc=img_np_hwc, show_face=show_face,
                            show_hands=show_hands, show_body=show_body)
            pose_frames.append(pose)

            face_bboxes.append(face_bbox)

            # flatten list
            # Append cropped faces and landmarks as individual tensors
            # if cropped_faces:
            #     for cropped_face in cropped_faces:
            #         cropped_faces_frames.append(torch.from_numpy(cropped_face.astype(np.float32) / 255))
            #     for cropped_face_lmk in cropped_face_lmks:
            #         cropped_faces_lmks_frames.append(torch.from_numpy(cropped_face_lmk.astype(np.float32) / 255))

            if cropped_faces:
                for cropped_face in cropped_faces:
                    cropped_faces_frames.append(cropped_face)
                for cropped_face_lmk in cropped_face_lmks:
                    cropped_faces_lmks_frames.append(cropped_face_lmk)

            pbar.update(1)

        pose_frames_np = np.array(pose_frames).astype(np.float32) / 255

        # if not cropped_faces_frames:
        #     # Create a single black image if no faces were detected
        #     black_image = torch.zeros((1, images.shape[1], images.shape[2], 3), dtype=torch.float32)
        #     cropped_faces_frames = black_image
        #     cropped_faces_lmks_frames = black_image
        # return (torch.from_numpy(pose_frames_np), cropped_faces_frames, cropped_faces_lmks_frames, face_bboxes,)

        pose_frames_np = np.array(pose_frames).astype(np.float32) / 255
        if cropped_faces_frames:
            cropped_faces_np = np.array(cropped_faces_frames, dtype=np.float32) / 255
            cropped_faces_lmks_frames_np = np.array(cropped_faces_lmks_frames, dtype=np.float32) / 255
        else:
            # black image
            cropped_faces_np = np.zeros((1, images.shape[1], images.shape[2], 3), dtype=np.float32) / 255
            cropped_faces_lmks_frames_np = np.zeros((1, images.shape[1], images.shape[2], 3), dtype=np.float32) / 255

        return (torch.from_numpy(pose_frames_np), torch.from_numpy(cropped_faces_np), torch.from_numpy(cropped_faces_lmks_frames_np), face_bboxes,)

class FacePaster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "cropped_faces": ("IMAGE",),
                "face_bboxes": ("FACE_BBOXES",),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "tensorrt"

    def main(self, original_images, cropped_faces, face_bboxes):
        pasted_images = original_images.clone()

        face_index = 0
        # Loop through each image in the batch
        for i, image_bboxes in enumerate(face_bboxes):
            # Loop through each face in the current image
            for (x1, y1), (x2, y2) in image_bboxes:
                if face_index < len(cropped_faces):
                    cropped_face = cropped_faces[face_index]
                    # Ensure the cropped face matches the bounding box size
                    resized_face = F.interpolate(cropped_face.unsqueeze(0).permute(0, 3, 1, 2), 
                                                 size=(y2-y1, x2-x1), 
                                                 mode='bilinear', 
                                                 align_corners=False)
                    resized_face = resized_face.squeeze(0).permute(1, 2, 0)
                    pasted_images[i, y1:y2, x1:x2, :] = resized_face
                    face_index += 1

        return (pasted_images, )

NODE_CLASS_MAPPINGS = {
    "DwposeTensorrt": DwposeTensorrt,
    "FacePaster": FacePaster,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DwposeTensorrt": "Dwpose Tensorrt",
    "FacePaster": "Face Paster",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
