## Fiji plugin

To ameliorate the difficulty of using our deep self-supervised learning-based method, we developed a user-friendly Fiji plugin, which is easy to install and convenient to use (has been tested on a desktop with Intel i9 CPU and 128G RAM). Researchers without expertise in computer science and machine learning can manage it in a very short time. 

<img src="https://github.com/cabooster/DeepCAD/blob/master/images/fiji.png" width="1000" align="middle">

### Install Fiji plugin
To avoid unnecessary troubles, the following steps are recommended for installation: 
1.  Download and install Fiji from the [[Fiji download page](https://imagej.net/Fiji/Downloads)]. Install the CSBDeep dependcy following the steps at [[CSBDeep in Fiji – Installation](https://github.com/CSBDeep/CSBDeep_website/wiki/CSBDeep-in-Fiji-%E2%80%93-Installation)]
2.  Download the packaged plugin file (.jar) from [[DeepCAD_Fiji/DeepCAD_Fiji_plugin](https://github.com/cabooster/DeepCAD/tree/master/DeepCAD_Fiji/DeepCAD_Fiji_plugin)]. 
3.  Install the plugin via **Fiji > Plugin > Install**. 

We provide lightweight data for test.  Please download the demo data (.tif) [[DataForTestPlugin](https://drive.google.com/drive/folders/1JVbuCwIxRKr4_NNOD7fY61NnVeCA2UnP)] and the pre-trained model (.zip and .yaml) [[ModelForTestPlugin](https://drive.google.com/drive/folders/14wSuMFhWKxW5Oq93GHxTsGixpB3T4lOL)]. 

### Use Fiji plugin

1.  Open Fiji.
2.  Open the calcium imaging stack to be denoised.
3.  Open the plugin at **Plugins > DeepCAD**. Six parameters will be shown on the panel (with default values and no changes are required unless necessary).
4.  Specify the pre-trained model using the '*Browse*' button (select the .zip file). 
5.  Click ‘OK’ and the denoised result will be displayed in another window after processing (processing time depends on the data size).
<img src="https://github.com/cabooster/DeepCAD/blob/master/images/parameter.PNG" width="700" align="middle">


### Train a customized model for your microscope

Because imaging systems and experiment conditions varies, a customized DeepCAD model trained on specified data is recommended for optimal performance. A Tensorflow implementation of DeepCAD compatible with the plugin is made publicly accessible at *[DeepCAD_Fiji/DeepCAD_tensorflow](https://github.com/cabooster/DeepCAD/tree/master/DeepCAD_Fiji/DeepCAD_tensorflow)*.
