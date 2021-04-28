import numbers
import torch
from ibug.roi_tanh_warping import *

class WarpingImageToDifferentSpace(object):
    """
    The warper class to project an image or a batch of images
        into different representation space like RoI Tanh Polar.

    Args:
        input_image_size: the size of input image, either an integer or an integer tuple of (H,W,)
        target_image_size: the size of output image, either an integer or an integer tuple of (H,W,)
        target_space: the target space to warp into.
            See WarpingImageToDifferentSpace.allowed_target_spaces from supported spaces
        is_training: is training or not. If is_training, add random offsets to RoI and also offsets to angular
        roi_ratio: the ratio of RoI region with respect to the whole image
        roi_offset_range: the offset range of RoI.
            Should satisfy max(roi_ratio)+2*max(abs(roi_offset_range))<=1.0.
            Only valid when is_training is True
        angular_offset_range: the angular offset range (in Radian). Only valid when is_training==True
        squeeze_output: Whether to squeeze the output
    """
    allowed_target_spaces = ("roi_tanh_polar","roi_tanh_circular", "roi_tanh")

    def __init__(
            self,
            input_image_size,
            target_image_size,
            target_space="roi_tanh_polar",
            is_training=True,
            roi_ratio=(0.8,0.8,),
            roi_offset_range=(-0.08,0.08,),
            angular_offset_range=(-0.35,0.35,),
            keep_aspect_ratio=True,
            squeeze_output: bool = True
    ):

        self.target_space=target_space

        # determine functions of warping and restoring
        if self.target_space=="roi_tanh_polar":
            self.warp_func = roi_tanh_polar_warp
            self.restore_func = roi_tanh_polar_restore
        elif self.target_space=="roi_tanh_circular":
            self.warp_func = roi_tanh_circular_warp
            self.restore_func = roi_tanh_circular_restore
        elif self.target_space=="roi_tanh":
            self.warp_func = roi_tanh_warp
            self.restore_func = roi_tanh_restore
        else:
            raise ValueError('Invalid target_space: {}. \n Allowed target_space:\n  {}'.format(
                target_space,
                WarpingImageToDifferentSpace.allowed_target_spaces
            ))

        if isinstance(target_image_size, numbers.Number):
            self.target_image_size = (
                int(target_image_size), int(target_image_size), )
        else:
            self.target_image_size = target_image_size

        if isinstance(input_image_size, numbers.Number):
            self.input_image_size = (
                int(input_image_size), int(input_image_size), )
        else:
            self.input_image_size = input_image_size

        self.is_training=is_training
        self.roi_ratio = roi_ratio

        self.original_roi_size = (
            int(self.input_image_size[0] * self.roi_ratio[0]),
            int(self.input_image_size[1] * self.roi_ratio[1]),
        )

        # original RoI position without offsets
        self.original_roi = torch.tensor((
            int(0.5 * self.input_image_size[1] - 0.5 * self.original_roi_size[1]),  # x1 (left width)
            int(0.5 * self.input_image_size[0] - 0.5 * self.original_roi_size[0]),  # y1 (top height)
            int(0.5 * self.input_image_size[1] + 0.5 * self.original_roi_size[1]),  # x2 (right width)
            int(0.5 * self.input_image_size[0] + 0.5 * self.original_roi_size[0]),  # y2 (bottom height)
        ))

        self.roi_offset_range = roi_offset_range
        self.angular_offset_range = angular_offset_range
        self.keep_aspect_ratio = keep_aspect_ratio

        self.roi_offset = (0.0,) * 4
        self.angular_offset = 0.0

        self.this_roi = None

        self.args = None
        self.key_args = None
        self.warped_image = None

        self.restore_args = None
        self.restore_key_args = None
        self.restored_image = None

        self.batch_image = None
        self.batch_size = 1

        self.squeeze_output = squeeze_output

    def __call__(self, image: torch.Tensor):
        """
        Args:
            image: images to be projected, should be a PyTorch Tensor with shape C*H*W or B*C*H*W

        Return:
            the projected images in the target space, a PyTorch Tensor of B*C*target_height*target_width
            (B=1 if input's shape is C*H*W)
        """
        img_shape = tuple(image.shape)

        if len(img_shape)==3:
            # C*H*W
            self.batch_size = 1
            self.batch_image = image.unsqueeze(0)
        elif len(img_shape)==4:
            # B*C*H*W
            self.batch_size = img_shape[0]
            self.batch_image = image
        else:
            raise ValueError(
                "Invalid input image. Should a Pytorch tensor with shape C*H*W or B*C*H*W")

        if self.is_training: # add random offsets to RoIs and angulars

            # sample 4*batch_size offset ratios
            self.roi_offset = torch.FloatTensor(4*self.batch_size).uniform_(
                self.roi_offset_range[0], self.roi_offset_range[1])

            self.roi_offset[::2] = self.roi_offset[::2] * self.input_image_size[1]  # offsets along width direction (x)
            self.roi_offset[1::2] = self.roi_offset[1::2] * self.input_image_size[0]  # offsets along height direction (y)
            self.roi_offset = self.roi_offset.int()

            # add offset to original roi boxes
            self.this_roi = self.original_roi.repeat(self.batch_size) + self.roi_offset
            self.this_roi = torch.reshape(self.this_roi, (self.batch_size, 4,)).int()

            # angular offsets
            self.angular_offset = torch.FloatTensor(self.batch_size).uniform_(
                self.angular_offset_range[0], self.angular_offset_range[1])

        else:
            self.this_roi = self.original_roi.repeat(self.batch_size)
            self.this_roi = torch.reshape(self.this_roi, (self.batch_size, 4,))
            self.angular_offset = 0.0

        self.args = (
            self.batch_image,   # images: torch.Tensor in B*C*H*W
            self.this_roi,    # rois: torch.Tensor in B*4
            self.target_image_size[1],   #  target_width: int
            self.target_image_size[0],   #  target_height: int
        )

        self.key_args = {
            "angular_offsets": self.angular_offset   # angular_offset: float or torch.Tensor
        }

        if self.target_space!="roi_tanh":
            # add "keep_aspect_ratio" key_args for none roi_tanh warping methods
            self.key_args["keep_aspect_ratio"] = self.keep_aspect_ratio

        self.warped_image = self.warp_func(*self.args, **self.key_args)

        if self.squeeze_output:
            return self.warped_image.squeeze()
        else:
            return self.warped_image


    def __repr__(self):
        return self.__class__.__name__

    def restore_warped_image(self):
        """
        Restore warped image back to original space

        Return:
            the warped image in the original space, which a PyTorch Tensor
        """
        self.restore_args = (self.warped_image,) + self.args[1:]
        self.restore_key_args = self.key_args
        self.restored_image = self.restore_func(
            *self.restore_args, **self.restore_key_args).squeeze()
        return self.restored_image

