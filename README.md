## Clothing detection using YOLOv3 in ModaNet dataset.

### Datasets

- ModaNet dataset: https://github.com/eBay/modanet
- Categories of clothing items include:
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

### Model

- YOLOv3 trained with Darknet framework: https://github.com/AlexeyAB/darknet

- To do inference use a pytorch implementation of YOLOv3: https://github.com/eriklindernoren/PyTorch-YOLOv3.

### Weights

All weights and config files are in https://drive.google.com/file/d/1BaWJ6j5HGC136h6f4kl_eo2LNPfjgyjq/view?usp=sharing

## Usage

- In <code>extraction_bb.py</code> , all the categories of clothing items detected in the input picture are saved in their respective directories. 
- Use <code>YOLOv3Predictor</code> class for YOLOv3 
