import os
import torch
import numpy as np
from typing import Union, Optional
from .utils.space_warper import WarpingImageToDifferentSpace
from .backbones import iresnet18, iresnet50, rtnet50
import cv2
from copy import deepcopy

__all__=["FaceEmbeddingPredictor"]


SPACE_TO_NAME = {
    None: "arcface_cartesian",
    "roi_tanh_polar": "arcface_roiTanhPolar",
    "roi_tanh": "arcface_roiTanh",
    "roi_tanh_circular": "arcface_roiTanhCircular"
}

class FaceEmbeddingPredictor(object):

    support_backbones = ("iresnet18", "iresnet50", "rtnet50",)
    support_spaces = (None, "roi_tanh_polar","roi_tanh_circular", "roi_tanh",)
    IMG_SIZE = (112, 112,)

    def __init__(
            self,
            backbone: str="iresnet18",
            project_to_space: Optional[str] = None,
            model_path: Optional[str] = None,
            device: Union[str, torch.device] = "cuda:0",
            roi_ratio: tuple = (0.8, 0.8,),
            keep_aspect_ratio: bool = True) -> None:

        self.backbone = backbone.lower().strip()
        if self.backbone not in FaceEmbeddingPredictor.support_backbones:
            raise ValueError("Do not support backbone: {} \n List of supported backbones: {}".format(
                self.backbone, FaceEmbeddingPredictor.support_backbones))

        if project_to_space is not None:
            self.project_to_space = project_to_space.strip()
        else:
            self.project_to_space = None

        if self.project_to_space not in FaceEmbeddingPredictor.support_spaces:
            raise ValueError("Do not support projection to space: {}".format(self.project_to_space))

        self.device = device
        self.roi_ratio = roi_ratio
        self.keep_aspect_ratio = keep_aspect_ratio

        if self.backbone == "iresnet18":
            self.net = iresnet18(False).to(self.device)
        elif self.backbone == "iresnet50":
            self.net = iresnet50(False).to(self.device)
        elif self.backbone == "rtnet50":
            self.net = rtnet50(False, dilated=False).to(self.device)
            assert self.project_to_space == "roi_tanh_polar", \
                "For rtnet backbone, project_to_space should be set to roi_tanh_polar"
        else:
            raise ValueError("Do not support backbone: {}".format(self.backbone))

        # print("Using backbone: {}".format(self.backbone))

        if model_path is not None:
            model_path = model_path.strip()
            assert os.path.exists(model_path), "Invalid model path: {}".format(model_path)
            self.model_path = os.path.realpath(model_path)
        else:
            model_name = SPACE_TO_NAME[self.project_to_space] + "_" + self.backbone + ".pth"
            self.model_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), 'weights', model_name))

        print("Loading weights from: {}".format(self.model_path))
        weight = torch.load(self.model_path,  map_location=self.device)

        self.net.load_state_dict(weight)
        self.net.eval()

        if self.project_to_space is not None:
            self._space_projector = WarpingImageToDifferentSpace(
                FaceEmbeddingPredictor.IMG_SIZE,
                FaceEmbeddingPredictor.IMG_SIZE,
                target_space=self.project_to_space,
                is_training=False,
                roi_ratio=self.roi_ratio,
                keep_aspect_ratio=self.keep_aspect_ratio,
                squeeze_output=False)  # keep the output as B*C*H*W
        else:
            self._space_projector = None

        self._embedding = None


    @torch.no_grad()
    def __call__(
            self,
            img: np.ndarray,
            bgr: bool = False,
            flip: bool = True,
            normalize_embedding: bool=True) -> np.ndarray:

        im_height, im_width, _ = img.shape

        if bgr:
            # bgr to rgb
            img = img[..., ::-1]

        if im_height != FaceEmbeddingPredictor.IMG_SIZE[0] \
                or im_width != FaceEmbeddingPredictor.IMG_SIZE[1]:
            img = cv2.resize(img, FaceEmbeddingPredictor.IMG_SIZE)

        if flip:
            img_f = deepcopy(img)
            img_f = img_f[:, ::-1, ...]  # horizontal flip image
            img = np.stack((img, img_f,),axis=0)
        else:
            img = np.expand_dims(img, axis=0)

        img = img.transpose(0, 3, 1, 2)
        img = torch.from_numpy(img.copy()).float().to(self.device)

        if self.project_to_space is not None:
            img = img / 255.0
            img = self._space_projector(img)  # project to a certain space
            img = (img - 0.5) / 0.5
        else:
            img = ((img / 255.0) - 0.5) / 0.5  # keep in original space (cartesian)

        if "iresnet" in self.backbone:
            net_out: torch.Tensor = self.net(img)
        else:
            net_out: torch.Tensor = self.net(img, self._space_projector.this_roi)

        if flip:
            # add together embeddings of original and flipped images
            net_out = torch.sum(net_out, dim=0)

        self._embedding = net_out.squeeze().detach().cpu().numpy()

        if normalize_embedding:
            self._embedding = self._embedding / np.linalg.norm(self._embedding)

        return self._embedding

