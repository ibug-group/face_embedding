import cv2
import numpy as np
from skimage import transform as trans

src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)

# In[66]:


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        assert image_size == 112
        src = arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def crop_face(im, bbox, extend=0.3, target_size=(112,112,)):
    """
    Crop a square facial region out of the image and resize it to a target size
    Args:
        im: the image to be cropped
        bbox: a [x1,y1,x2,y2] bounding box where 0<=x1<x2 and 0<=y1<y2
        extend: the ratio to extend the cropped region. A bigger extend will lead to a larger cropping region.
        target_size: the target size of the output cropped facial images

    Returns:
        a cropped image of target size
    """
    x1, y1, x2, y2 = bbox

    cropWidth = round(x2 - x1)
    cropHeight = round(y2 - y1)
    cropLength = (cropWidth + cropHeight) / 2.0

    cenPoint = [round(x1 + 0.5 * cropWidth), round(y1 + 0.5 * cropHeight)]

    # find square region to crop
    x1n = cenPoint[0] - round((1 + extend) * cropLength * 0.5)
    y1n = cenPoint[1] - round((1 + extend) * cropLength * 0.5)
    x2n = cenPoint[0] + round((1 + extend) * cropLength * 0.5)
    y2n = cenPoint[1] + round((1 + extend) * cropLength * 0.5)

    # prevent going out of image
    x1n = max(0, x1n)
    y1n = max(0, y1n)
    x2n = min(x2n, im.shape[1])
    y2n = min(y2n, im.shape[0])

    cropped = im[y1n:y2n, x1n:x2n, ...]
    cropped = cv2.resize(cropped, target_size)

    return cropped

