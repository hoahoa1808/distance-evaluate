import numpy as np


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm
    return output


def create_id(cls, img_ids):
    ids = []
    for c, img_id in zip(cls, img_ids):
        face_id = int((int(c)+1e6)*1e4+int(img_id))
        ids.append(face_id)

    return ids