import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
import sys
from typing import Union
import argparse

# plot one image or a list of images
def imshow(img,normalize_range=None,cmap='Oranges',clip=True):

    if type(img) is np.ndarray:
        img = [img]

    for this_img in img:
        if normalize_range is not None:
            norm=matplotlib.colors.Normalize(
                vmin=np.min(normalize_range),
                vmax=np.max(normalize_range),
                clip=clip)
            mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            this_img = mapper.to_rgba(this_img)
        else:
            this_img = np.squeeze(this_img).astype(np.uint8)

        plt.figure()
        plt.imshow(this_img)
    plt.show()


# print on the same line (overwriting previous contents)
def print_overwrite(print_str):
   print(print_str, end='')
   sys.stdout.flush()
   print('\r', end='')


# convert a large number to human readable string
def human_format(num, decimal='.2f'):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    format_str = '%'+decimal+'%s'
    return format_str % (num, ['', 'K', 'M', 'B', 'T', 'P'][magnitude])


def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

def get_projection_options(input_args: Union[argparse.Namespace, dict]) -> dict:
    """
    Get arguments for initialising WarpingImageToDifferentSpace class from the argparse.Namespace

    Parameters
    ----------
    input_args: the argparse.Namespace object or a dict

    Returns
    -------
    the packed arguments for for initialising WarpingImageToDifferentSpace class
    """
    roi_ratio = tuple(
        float(item) for item in input_args.roi_ratio.strip().split(","))
    roi_offset_range = tuple(
        float(item) for item in input_args.roi_offset_range.strip().split(","))
    angular_offset_range = tuple(
        float(item) for item in input_args.angular_offset_range.strip().split(","))
    keep_aspect_ratio = not input_args.ignore_aspect_ratio

    target_args = {
        "roi_ratio": roi_ratio,
        "roi_offset_range": roi_offset_range,
        "angular_offset_range": angular_offset_range,
        "keep_aspect_ratio": keep_aspect_ratio
    }

    return target_args

