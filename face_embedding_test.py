import os
import cv2
import time
import torch
from argparse import ArgumentParser
import numpy as np
from ibug.face_embedding import FaceEmbeddingPredictor
from ibug.face_detection import RetinaFacePredictor
from ibug.face_embedding.utils.preprocessing import crop_face, norm_crop

path1 = "samples/Niki_DeLoach_1.png"
path2 = "samples/Peri_Gilpin_1.png"

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input1', help='Path of verification image 1', default=path1)
    parser.add_argument('--input2', help='Path of verification image 2', default=path2)
    parser.add_argument('--threshold', '-t', help='Face verification threshold (default: 1.66)',
                        type=float, default=1.66)
    parser.add_argument('--backbone', '-b',
                        help='Backbone for embedding. Supported backbones: {} (default: iresnet18)'.format(
                            FaceEmbeddingPredictor.support_backbones),
                        default='iresnet18')
    parser.add_argument('--project_to_space', '-p',
                        help='The space to project the image into. Supported spaces: {} (default: None)'.format(
                            FaceEmbeddingPredictor.support_spaces[1:]),
                        default=None)
    parser.add_argument('--align_face', '-a',
                        type=int,
                        help='If 1, will align the face based on 5 landmarks. If 0, will not align. (default: 1)',
                        default=1)
    parser.add_argument('--flip', '-f',
                        type=int,
                        help='If 1, will additionally flip image during embedding. If 0, will not. (default: 1)',
                        default=1)
    parser.add_argument('--model_path', '-m',
                        help='Weights to load. If not specified, will load weights from pre-defined path. (default: None)',
                        default=None)
    parser.add_argument('--device', '-d', help='Device to be used by the model (default: cuda:0)',
                        default='cuda:0')
    args = parser.parse_args()
    return args


def main():

    args = get_args()

    image_size = FaceEmbeddingPredictor.IMG_SIZE

    # read images
    img1 = cv2.imread(args.input1.strip())
    img2 = cv2.imread(args.input2.strip())

    print("Image 1: {}".format(args.input1))
    print("Image 2: {}".format(args.input2))
    print("Verifying image 1 and 2")

    face_detector = RetinaFacePredictor(
        threshold=0.8,
        device=args.device.strip(),
        model=RetinaFacePredictor.get_model('resnet50'))

    det1 = face_detector(img1, rgb=False)[0].astype(int)
    bbox1 = det1[0:4]  # bounding box 1
    ldm1 = det1[5:].reshape((5,2,))   # landmarks 1

    det2 = face_detector(img2, rgb=False)[0].astype(int)
    bbox2 = det2[0:4]   # bounding box 2
    ldm2 = det2[5:].reshape((5,2,))   # landmarks 2

    if bool(args.align_face):
        # crop the faces and align them using landmarks
        print('Faces will be aligned')
        face1 = norm_crop(img1, ldm1, image_size=image_size[0])
        face2 = norm_crop(img2, ldm2, image_size=image_size[0])
    else:
        print('Faces will not be aligned')
        # crop the faces, but do not align them
        face1 = crop_face(img1, bbox1, extend=0.2, target_size=image_size)
        face2 = crop_face(img2, bbox2, extend=0.2, target_size=image_size)

    print('Loading embedding predictor with backbone: {}'.format(args.backbone))
    embedding_predictor = FaceEmbeddingPredictor(
        backbone=args.backbone,
        project_to_space=args.project_to_space,
        model_path=args.model_path,
        device=args.device)

    if args.project_to_space is not None:
        print("Projecting images into space: {} ".format(args.project_to_space))
    print('Extracting face embeddings ...')
    start_time = time.time()
    embeddings1 = embedding_predictor(face1, bgr=True, flip=bool(args.flip), normalize_embedding=True)
    embeddings2 = embedding_predictor(face2, bgr=True, flip=bool(args.flip), normalize_embedding=True)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff))

    end_time = time.time()
    print("Verification time: {:.3f} sec".format(end_time - start_time))

    if dist < args.threshold:
        print("Embedding distance ({:.2f}) < Threshold ({:.2f})".format(dist, args.threshold))
        print("Image 1 and 2 are from the same person\n")
    else:
        print("Embedding distance ({:.2f}) > Threshold ({:.2f})".format(dist, args.threshold))
        print("Image 1 and 2 are from different persons\n")


if __name__ == '__main__':
    main()