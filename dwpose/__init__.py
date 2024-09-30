# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from . import util
from .wholebody import Wholebody

def draw_pose(pose, image_np_hwc):
    H,W = image_np_hwc.shape[:2]
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas

def draw_face_lmks(faces, image_np_hwc):
    H,W = image_np_hwc.shape[:2]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = util.draw_facepose(canvas, faces, 1)
    return canvas

class DWposeDetector:
    def __init__(self,engine,engine2):
        self.pose_estimation = Wholebody(engine,engine2)

    def __call__(self, image_np_hwc, show_face, show_hands, show_body):
        image_np_hwc= image_np_hwc.copy()

        
        H, W, C = image_np_hwc.shape
        
        candidate, subset = self.pose_estimation(image_np_hwc)

        nums, keys, locs = candidate.shape
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        body = candidate[:,:18].copy()
        body = body.reshape(nums*18, locs)
        score = subset[:,:18]
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] > 0.3:
                    score[i][j] = int(18*i+j)
                else:
                    score[i][j] = -1

        un_visible = subset<0.3
        candidate[un_visible] = -1

        # foot = candidate[:,18:24]

        faces = candidate[:,24:92]

        left_hands = candidate[:,92:113]
        right_hands = candidate[:,113:]

        hands = np.vstack([left_hands, right_hands])

        bodies = dict(candidate=body, subset=score)
        
        pose = dict(bodies=bodies if show_body else {'candidate':[], 'subset':[]}, faces=faces if show_face else [], hands=hands if show_hands else [])

        face_bbox = util.convert_face_lmks_to_bbox(faces, H, W)

        cropped_faces = util.crop_face_by_bbox(image_np_hwc, face_bbox)

        pose = draw_pose(pose, image_np_hwc)

        face_lmks_img = draw_face_lmks(faces, image_np_hwc)
        cropped_face_lmks = util.crop_face_by_bbox(face_lmks_img, face_bbox)
        
        return pose, face_bbox, cropped_faces, cropped_face_lmks
