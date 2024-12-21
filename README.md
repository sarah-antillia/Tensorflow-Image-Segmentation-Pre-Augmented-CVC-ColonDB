<h2>Tensorflow-Image-Segmentation-Pre-Augmented-CVC-ColonDB (2024/12/22)</h2>

This is the first experiment of Image Segmentation for CVC-ColonDB 
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
 a pre-augmented <a href="https://drive.google.com/file/d/1e3HA3_Eo0rV2kp89tXQ1vVeDZCHmn2O9/view?usp=sharing">
CVC-ColonDB-ImageMask-Dataset.zip</a>, which was derived by us from 
<a href="https://www.kaggle.com/datasets/longvil/cvc-colondb"><b>CVC-ColonDB</b></a> on Kaggle website.
<br>
<br>
<b>Data Augmentation Strategy:</b><br>
 To address the limited size of the original CVC-ColonDB, which contains 380 images and their corresponding masks,, 
 we employed <a href="./generator/ImageMaskDatasetGenerator.py">an offline augmentation tool</a> to generate a pre-augmented dataset, which supports the following augmentation methods.
<li>Vertical flip</li>
<li>Horizontal flip</li>
<li>Rotation</li>
<li>Shrinks</li>
<li>Shears</li> 
<li>Deformation</li>
<li>Distortion</li>
<li>Barrel distortion</li>
<li>Pincushion distortion</li>
<br>
Please see also the following tools <br>
<li><a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a></li>
<li><a href="https://github.com/sarah-antillia/Image-Distortion-Tool">Image-Distortion-Tool</a></li>
<li><a href="https://github.com/sarah-antillia/Barrel-Image-Distortion-Tool">Barrel-Image-Distortion-Tool</a></li>

<br>

<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/images/102.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/masks/102.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test_output/102.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/images/204.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/masks/204.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test_output/204.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/images/340.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/masks/340.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test_output/340.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this CVC-ColonDBSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been taken from the following kaggle web-site:<br>
<a href="https://www.kaggle.com/datasets/longvil/cvc-colondb"><b>CVC-ColonDB</b></a>.
<br><br>
You may also download CVC-ColonDB from the following web site.<br>
<a href="http://vi.cvc.uab.es/colon-qa/cvccolondb/">CVC colon DB</a>
<br><br>
<b>Citation</b><br>
Jorge Bernal, F. Javier Sanchez, & Fernando Vilariño. (2012). <br>
“Towards Automatic Polyp Detection with a Polyp Appearance Model “ .<br>
 Pattern Recognition, 45(9), 3166–3182.
<br>
(C) 2010 - 2024 Machine Vision Group.<br>
Computer Vision Centre. Edificio O 08193 Bellaterra, Barcelona (Spain).<br>
<br>
<b>License</b>: Unknown
<br>
<h3>
<a id="2">
2 CVC-ColonDB ImageMask Dataset
</a>
</h3>
 If you would like to train this CVC-ColonDB Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1e3HA3_Eo0rV2kp89tXQ1vVeDZCHmn2O9/view?usp=sharing">
CVC-ColonDB-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─CVC-ColonDB
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
On the derivation of this dataset, please refer to the following Python scripts:<br>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py.</a></li>
<br>
The is a pre-augmented dataset generated by the ImageMaskDatasetGenerator.py.<br>
<br>
<b>CVC-ColonDB Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/CVC-ColonDB_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough to use for a training set of our segmentation model.
<!-- Therefore, we enabled our online augmentation tool in the training process.
-->
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained CVC-ColonDBTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDBand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Enabled Batch Normalization.<br>
Defined a small <b>base_filters:16 </b> and large <b>base_kernels:(9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
normalization  = True
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 7
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.You may train this model by setting this generator parameter to True. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>


<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 36 by EarlyStoppling callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/asset/train_console_output_at_epoch_36.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for CVC-ColonDB.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/asset/evaluate_console_output_at_epoch_36.png" width="720" height="auto">
<br><br>Image-Segmentation-CVC-ColonDB

<a href="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this CVC-ColonDB/test was not so low, and dice_coef so high as shown below.
<br>
<pre>
loss,0.1336
dice_coef,0.8207
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for CVC-ColonDB.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/images/160.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/masks/160.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test_output/160.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/images/246.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/masks/246.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test_output/246.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/images/259.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/masks/259.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test_output/259.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/images/285.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/masks/285.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test_output/285.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/images/316.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/masks/316.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test_output/316.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/images/338.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test/masks/338.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/CVC-ColonDB/mini_test_output/338.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>

<b>1. Automated Detection of Colorectal Polyp Utilizing Deep Learning Methods With Explainable AI</b><br>
Faysal Ahamed, Rabiul Islam, Nahiduzzaman, Jawadul Karim, Mohamed Arselene Ayari, Amith Khandakar<br>

<a href="https://ieeexplore.ieee.org/document/10534764">https://ieeexplore.ieee.org/document/10534764</a><br>
<br>
<b>2. Rethinking the transfer learning for FCN based polyp segmentation in colonoscopy</b><br>
Yan Wen, Lei Zhang1, Xiangli Meng and Xujiong Ye<br>
<a href="https://arxiv.org/pdf/2211.02416">https://arxiv.org/pdf/2211.02416</a>

