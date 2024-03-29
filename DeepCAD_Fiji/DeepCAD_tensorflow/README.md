## Tensorflow implementation compatible with Fiji plugin

### Directory structure
```
DeepCAD_tensorflow #Tensorflow implementation compatible with Fiji plugin#
|---basic_ops.py
|---train.py
|---network.py
|---script.py
|---test_pb.py
|---data_process.py
|---datasets
|---|---qwd_7 #project_name#
|---|---|---train_raw.tif #raw data for train#
|---DeepCAD_model
|---|---qwd_7_20201115-0913
|---|---|--- #pb model#
|---results
|---|---#Results of training process and final test#
```

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
$ pip install tensorflow-gpu==1.4.0
```

### Training

```
$ source activate tensorflow
$ python main.py --GPU 0 --img_h 64 --img_w 64 --img_s 320 --train_epochs 30 --datasets_folder DataForPytorch --normalize_factor 1 --lr 0.00005 --train_datasets_size 1000
```

Parameters can be modified as required.

```
$ python main.py --GPU #GPU index# --img_h #stack height# --img_w #stack width# --img_s #stack length# --train_epochs #training epoch number# --datasets_folder #project name#
```

The pre-trained model is saved at *DeepCAD_Fiji/DeepCAD_tensorflow/DeepCAD_model/*. 

#### Test

Run the script.py (test part) to begin your test. Parameters saved in the .yaml file will be automatically loaded.

```
$ source activate pytorch
$ python test_pb.py --GPU 3 --denoise_model ModelForTestPlugin --datasets_folder DataForPytorch --model_name 25_1000 --test_datasize 500
```

Parameters can be modified  as required.

```
$ os.system('python test.py --denoise_model #model name# --test_datasize #the number of images used for test#')
```
