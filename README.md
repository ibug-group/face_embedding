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
* Organize the training data as "ImageFolder" format that can be read by `torchvision.datasets.ImageFolder`. Example "ImageFolder" structure:
```
$data_root/Adam_Brody/xxx.png
$data_root/Adam_Brody/xxy.png
$data_root/Adam_Brody/[...]/xxz.png
...
$data_root/Brendan_Fraser/123.png
$data_root/Brendan_Fraser/nsdf3.png
$data_root/Brendan_Fraser/[...]/asd932_.png
```
`$data_root` stands for the dataset root directory.

* Download the [verification data](https://drive.google.com/file/d/116CLHSfV_lUtXIeKvaJ0M0ycZ2dBv9pU/view?usp=sharing) and unzip it to a directory `$ver_dir`

* In `./train.sh`, replace `$data_root` and `$ver_dir` with training and verification data paths, and also change `$output_dir` to the path of saving training data. Then run it:
```
bash train.sh
```


## References
\[1\] Deng, Jiankang, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. "Arcface: Additive angular margin loss for deep face recognition." In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 4690-4699. 2019.

\[2\] Lin, Yiming, Jie Shen, Yujiang Wang, and Maja Pantic. "RoI Tanh-polar Transformer Network for Face Parsing in the Wild." _arXiv preprint arXiv:2102.02717._ 2021.

\[3\] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning for image recognition." In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pp. 770-778. 2016.