# DeepCAD: Deep self-supervised learning for Calcium imaging denoising

<img src="images/logo.PNG" width="900" align="middle">

## Contents

- [Overview](#overview)
- [Directory structure](#directory-structure)
- [Pytorch code](#pytorch-code)
- [Fiji plugin](#fiji-plugin)
- [Results](#results)
- [License](./LICENSE)
- [Citation](#citation)

## Overview

<img src="images/schematic.png" width="400" align="right">

Calcium imaging is inherently susceptible to detection noise especially when imaging with high frame rate or under low excitation dosage. However, calcium transients are highly dynamic, non-repetitive activities and a firing pattern cannot be captured twice. Clean images for supervised training of deep neural networks are not accessible. Here, we present DeepCAD, a **deep** self-supervised learning-based method for **ca**lcium imaging **d**enoising. Using our method, detection noise can be effectively removed and the accuracy of neuron extraction and spike inference can be highly improved.

DeepCAD is based on the insight that a deep learning network for image denoising can achieve satisfactory convergence even the target image used for training is another corrupted sampling of the same scene [[paper link]](https://arxiv.org/abs/1803.04189). We explored the temporal redundancy of calcium imaging and found that any two consecutive frames can be regarded as two independent samplings of the same underlying firing pattern. A single low-SNR stack is sufficient to be a complete training set for DeepCAD. Furthermore, to boost its performance on 3D temporal stacks, the input and output data are designed to be 3D volumes rather than 2D frames to fully incorporate the abundant information along time axis.

For more details, please see the companion paper where the method appeared: 
["*Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised learning*".](https://www.biorxiv.org/content/10.1101/2020.11.16.383984v1)

## Directory structure

```
DeepCAD
|---DeepCAD_pytorch #Pytorch implementation of DeepCAD#
|---|---train.py
|---|---test.py
|---|---script.py
|---|---network.py
|---|---model_3DUnet.py
|---|---data_process.py
|---|---buildingblocks.py
|---|---utils.py
|---|---datasets
|---|---|---DataForPytorch #project_name#
|---|---|---|---data.tif
|---|---pth
|---|---|---ModelForPytorch
|---|---|---|---model.pth
|---|---results
|---|---|--- # Intermediate and final results#
|---DeepCAD_Fiji
|---|---DeepCAD_Fiji_plugin
|---|---|---DeepCAD-0.3.0 #executable jar file/.jar#
|---|---DeepCAD_java #java source code of DeepCAD Fiji plugin#
|---|---DeepCAD_tensorflow #Tensorflow implementation compatible with Fiji plugin#
```
- **DeepCAD_pytorch** is the [Pytorch](https://pytorch.org/) implementation of DeepCAD.
- **DeepCAD_Fiji** is a user-friendly [Fiji](https://imagej.net/Fiji) plugin. This plugin is easy to install and convenient to use. Researchers without expertise in computer science and machine learning can learn to use it in a very short time. 
  - **DeepCAD_Fiji_plugin** contains the executable .jar file that can be installed on Fiji. 
  - **DeepCAD_java** is the java source code of our Fiji plugin based on [CSBDeep](https://csbdeep.bioimagecomputing.com). 
  - **DeepCAD_tensorflow** is the [Tensorflow](https://www.tensorflow.org/) implementation of DeepCAD, which is used for training models compatible with the Fiji plugin. 

## Pytorch code

### Environment 

* Ubuntu 16.04 
* Python 3.6
* Pytorch >= 1.3.1
* NVIDIA GPU (24 GB Memory) + CUDA

### Environment configuration

Open the terminal of ubuntu system.

* Install Pytorch

```
$ conda create -n deepcad python=3.6
$ source activate deepcad
$ pip install torch==1.3.1
$ conda install pytorch torchvision cudatoolkit -c pytorch
```

* Install other dependencies

```
$ conda install -c anaconda matplotlib opencv scikit-learn scikit-image
$ conda install -c conda-forge h5py pyyaml tensorboardx tifffile
```

### Training

Download the demo data(.tif file) [[DataForPytorch](https://drive.google.com/drive/folders/1w9v1SrEkmvZal5LH79HloHhz6VXSPfI_)] and put it into *DeepCAD_pytorch/datasets/DataForPytorch.*.

Run the **script.py (python script.py train)* to begin your train.

```
$ source activate deepcad
$ python script.py train
```

Parameters can be modified  as required in **script.py**. If GPU runs out of memory, you can use smaller `img_h`, `img_w`, `img_s`.

```
$ os.system('python train.py --datasets_folder #A folder containing files fortraining# --img_h #image height# --img_w #image width# --img_s #stack length# --gap_h #stack gap height# --gap_w #stack gap width# --gap_s #stack gap length# --n_epochs #the number of training epochs# --GPU #GPU index# --normalize_factor #normalization factor# --train_datasets_size #datasets size for training# --select_img_num #the number of images used for training#')
```

### Test

Download our pre-trained model (.pth file and .yaml file) [[ModelForPytorch](https://drive.google.com/drive/folders/12LEFsAopTolaRyRpJtFpzOYH3tBZMGUP)] and put it into *DeepCAD_pytorch/pth/ModelForPytorch*.

Run the **script.py (python script.py test)** to begin your test. Parameters saved in the .yaml file will be automatically loaded. If GPU runs out of memory, you can use smaller `img_h`, `img_w`, `img_s`.

```
$ source activate deepcad
$ python script.py test
```

Parameters can be modified  as required in **script.py**. All models in the `--denoise_model` folder will be tested and manual inspection should be made for **model screening**.

```
$ os.system('python test.py --denoise_model #A folder containing models to be tested# --datasets_folder #A folder containing files to be tested# --test_datasize #dataset size to be tested#')
```

## Fiji plugin

To ameliorate the difficulty of using our deep self-supervised learning-based method, we developed a user-friendly Fiji plugin, which is easy to install and convenient to use (has been tested on a Windows desktop with Intel i9 CPU and 128G RAM). Researchers without expertise in computer science and machine learning can manage it in a very short time. **Tutorials** on installing and using the plugin has been moved to [**this page**](https://github.com/cabooster/DeepCAD/tree/master/DeepCAD_Fiji).

<img src="https://github.com/cabooster/DeepCAD/blob/master/images/fiji.png" width="1000" align="middle">


## Results

### 1. The performance of DeepCAD on denoising two-photon calcium imaging of neurite activities.

<img src="images/dendrite.png" width="800" align="middle">

### 2. The performance of DeepCAD on denoising two-photon calcium imaging of large neuronal populations.

<img src="images/soma.png" width="800" align="middle">

### 3. Cross-system validation.

<img src="images/cross-system.png" width="800" align="middle">

Denoising performance of DeepCAD on three two-photon laser-scanning microscopes (2PLSMs) with different system setups. **Our system** was equipped with alkali PMTs (PMT1001, Thorlabs) and a 25×/1.05 NA commercial objective (XLPLN25XWMP2, Olympus). The **standard 2PLSM** was equipped with a GaAsP PMT (H10770PA-40, Hamamatsu) and a 25×/1.05 NA commercial objective (XLPLN25XWMP2, Olympus). The **two-photon mesoscope** was equipped with a GaAsP PMT (H11706-40, Hamamatsu) and a 2.3×/0.6 NA custom objective. The same pre-trained model was used for processing these data. 

## Citation

If you use this code please cite the companion paper where the original method appeared: 

["*Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised learning*".](https://www.biorxiv.org/content/10.1101/2020.11.16.383984v1)
