# Clothing detection using YOLOv3

Outfit item detection implemented using YOLOv3 object detection model trained on Modanet clothing dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kritanjalijain/Clothing_Detection_YOLO/blob/main/YOLOv3.ipynb)

## Project Description

Given an input image, the model detects one or more clothing item(s) categories and draws bounding boxes along with their prediction confidence score. Then it crops the items within the bounding boxes and saves them in the respective category directory. 

### About the Dataset
The weights have been trained on the [ModaNet dataset](https://github.com/eBay/modanet). 
The ModaNet dataset provides a large-scale street fashion image dataset with rich annotations, including polygonal/pixel-wise segmentation masks, bounding boxes. 
It consists of a training set of 52,377 images and a validation set of 2,799 images. This split ensures that each category from the validation set contains at least 500 instances, so that the validation accuracy is reliable. 

It contains 13 meta categories, where each meta category groups highly related categories to reduce the ambiguity in the annotation process. 
The 13 meta categories of clothing items include:
  - bag
  - belt
  - boots
  - footwear
  - outer
  - dress
  - sunglasses
  - pants
  - top
  - shorts
  - skirt
  - headwear
  - scarf/tie

### About the Model 

#### Training
YOLOv3 trained with Darknet framework: https://github.com/AlexeyAB/darknet. 

The training dataset is downloaded, extracted and stored in a sub-directory wherein Darknet is installed. Next, the label files required by Darknet were generated. Darknet needs a `.txt` file for each image with a line for each ground truth object in the image that contains x, y, width, and height which are relative to the image's width and height. Thus, firstly a script is written to parse all files to Yolo format, and save it to the same folder as that of the training images. Then, the
`modanet.data` config file is modified to point to the data. Additionally, a `.names` file is created to provide darknet with a listing of the names of classes of objects i.e., the 13 categories of the Modanet training dataset. 

Next, a weights file is prepared since the darknet binary will use each weights file to initialize the values in each cell in each filter in the YOLOv3 convolutional neural network when either training the network or forward propagating an image frame through the network. The [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74) is an appropriately formatted weights file that is used here to initialize the weights in a darknet YOLOv3 convolutional neural network. It has been used for initializing the weights of a darknet YOLOv3 convolutional neural network to detect objects similar to objects in one of the one-hundred million images in the ImageNet database.

More information on training using Darknet can be found here: https://pjreddie.com/darknet/yolo/

#### Inference
For inference, use a pytorch implementation of YOLOv3: https://github.com/eriklindernoren/PyTorch-YOLOv3

#### Weights

Weights and config files are in https://drive.google.com/file/d/1BaWJ6j5HGC136h6f4kl_eo2LNPfjgyjq/view?usp=sharing

## Testing 

- The script [`YOLOv3.py`](https://github.com/kritanjalijain/Clothing_Detection_YOLO/tree/main/predictors) contains the class <code>YOLOv3Predictor</code> for YOLOv3.
- <code>extraction_bb.py</code> , detects all the categories of clothing items in the input picture, crops their ROI and saves in their respective directories while `extract_top.py` does the same but only for the category topwear.
- `new_image_demo.py` detects the outfit items, draws bounding boxes and labels them along with their confidence scores.

## Results

The results on a few test images are visualised using `new_image_demo.py` and stored in the `output` directory.

<img src="https://github.com/kritanjalijain/Clothing_Detection_YOLO/blob/main/output/output-test_test1_yolo_modanet.jpg" height= 400 width=300 align=left>
<img src="https://github.com/kritanjalijain/Clothing_Detection_YOLO/blob/main/output/ouput-test_File_003_yolo_modanet.jpg" height= 400 width=280 align=center>


## Built With
* Languages - python
* Main Libraries - PyTorch, Torchvision, OpenCV

## Setup and Installation
* Clone the repository 
``` 
git clone https://github.com/kritanjalijain/Clothing_Detection_YOLO.git
```
* Change to working directory
```
cd Clothing_Detection_YOLO
```
* Install all dependencies (preferrably in a virtual env)
```
pip install -r requirements.txt
```
* For the demo, run
```
python new_image_demo.py
```


