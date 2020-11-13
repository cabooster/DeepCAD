## Fiji plugin

To ameliorate the difficulty of using our deep self-supervised learning-based method, we developed a user-friendly Fiji plugin, which is easy to install and convenient to use. Researchers without expertise in computer science and machine learning can manage it in a very short time. 

<img src="https://github.com/cabooster/DeepCAD/blob/master/images/fiji.png" width="900" align="middle">

### Install Fiji plugin

Download the packaged plugin file (.jar) from *[DeepCAD_Fiji/DeepCAD_Fiji_plugin](https://github.com/cabooster/DeepCAD/tree/master/DeepCAD_Fiji/DeepCAD_Fiji_plugin)*. Install the plugin via **Fiji > Plugin > Install**.

### Use Fiji plugin

1.  Open Fiji.

2.  Open the calcium imaging stack to be denoised.

3.  Open the plugin at **Plugins > DeepCAD**.

4.  Select the pre-trained model and set six parameters on the panel (with default values and no changes are required unless necessary).

<img src="https://github.com/cabooster/DeepCAD/blob/master/images/parameter.PNG" width="800" align="middle">

5.  Click ‘OK’ and the denoised result will be displayed in another window after several minutes (depends on your data size).

### Train a customized model for your microscope

Because imaging systems and experiment conditions varies, a customized DeepCAD model trained on specified data is recommended for optimal performance. A Tensorflow implementation of DeepCAD compatible with the plugin is made publicly accessible at **DeepCAD_Fiji/DeepCAD_tensorflow**.