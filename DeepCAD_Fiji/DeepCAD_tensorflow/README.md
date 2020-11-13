## Tensorflow implementation compatible with Fiji

### Environment 

* Ubuntu 16.04 
* Python 3.6
* Tensorflow 1.4.0
* NVIDIA GPU + CUDA

### Environment configuration

Open the terminal of ubuntu system.

* Create anaconda environment

```
$ conda create -n tensorflow python=3.6
```

* Install Tensorflow

```
$ source activate tensorflow
$ conda install tensorflow-gpu=1.4
```

#### Training

```
$ source activate tensorflow
$ os.system('python main2.py --GPU 0 --img_h 64 --img_w 64 --img_s 320 --train_epochs 30 --datasets_folder DataForPytorch --normalize_factor 1 --lr 0.00005 --train_datasets_size 1000')
```

Parameters can be modified as required.

```
$ python main.py --GPU #GPU index# --img_h #stack height# --img_w #stack width# --img_s #stack length# --train_epochs #training epoch number# --datasets_folder #project name#
```

The pre-trained model is saved at *DeepCAD_Fiji/DeepCAD_tensorflow/DeepCAD_model/*. 
