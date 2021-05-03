# ibug.face_embedding
A collection of pre-trained face embedding models based on ArcFace [1] and RoI Tanh Polar transformer [2]. Both embedding predictor and training codes are included. 

The embedding predictor supports the following backbones (iResNet is a variant of ResNet [3]):
* iResNet-18  
* iResNet-50 
* RTNet-50 [2] (pretrained model will be updated) 

Supporting warping spaces:
* Cartesian (No projection)
* RoI Tanh Polar 
* RoI Tanh
* RoI Tanh Circular 

## Prerequisites
* Tested with python 3.9.4, cuda 11.1.1, cudnn 8.1.0.77, pytorch 1.8.1, torchvision 0.9.1 
* All other dependency can be installed using requirements.txt `$pip install -r requirements.txt`

Let $ROOT denotes this project's root directory where this README resides

## How to Test 
* ibug.face_detection is required for testing the embedding predictor
* Download the [pretrained models](https://drive.google.com/file/d/13pwzWiQ6VEZ__VfnXFnyByKp-csYanQB/view?usp=sharing), extract and place as following:
```
├── $ROOT
│   ├── ibug
│       └── face_embedding
│           └── weights
│               └── arcface_cartesian_iresnet18.pth
│               ...
```
* Run the testing script:
```
cd $ROOT
bash test.sh
```

## How to Train
* Download _MS1M-ArcFace_ dataset from [ArcFace Dataset-Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo).
* Extract the downloaded dataset. In `$ROOT/ibug/face_embedding/utils/train_config.py`, modify `config.rec` at line 15 to be the root directory of the extracted dataset. 
* An example training session can be run:
```
cd $ROOT
bash train.sh
```
The output data will be saved in `$ROOT/../fr_snapshots/arcface_emore_root`, which can be changed by modifying `config.output` in the same `train_config.py`. 

## References
\[1\] Deng, Jiankang, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. "Arcface: Additive angular margin loss for deep face recognition." In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 4690-4699. 2019.

\[2\] Lin, Yiming, Jie Shen, Yujiang Wang, and Maja Pantic. "RoI Tanh-polar Transformer Network for Face Parsing in the Wild." _arXiv preprint arXiv:2102.02717._ 2021.

\[3\] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning for image recognition." In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pp. 770-778. 2016.