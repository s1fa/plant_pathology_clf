---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: labels
    sequence:
      class_label:
        names:
          '0': complex
          '1': frog_eye_leaf_spot
          '2': healthy
          '3': powdery_mildew
          '4': rust
          '5': scab
  - name: label_names
    sequence: string
  - name: image_id
    dtype: string
  splits:
  - name: train
    num_bytes: 14557242028.669252
    num_examples: 16768
  - name: validation
    num_bytes: 1603451702.490748
    num_examples: 1864
  download_size: 16094435250
  dataset_size: 16160693731.16
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
license: apache-2.0
task_categories:
- image-classification
tags:
- multi-label
pretty_name: PlantPathology-Challenge-2021-FGVC8
size_categories:
- 10K<n<100K
---
## Description
Dataset from the Plant Pathology 2021 (FGVC8) Challenge.

'
For Plant Pathology 2021-FGVC8, we have significantly increased the number of foliar disease images and added additional disease categories. This year’s dataset contains approximately 23,000 high-quality RGB images of apple foliar diseases, including a large expert-annotated disease dataset. This dataset reflects real field scenarios by representing non-homogeneous backgrounds of leaf images taken at different maturity stages and at different times of day under different focal camera settings.
'

The original dataset has one train split and a test split that was hidden for the challenge. I have taken 10% of train for a validation, using stratified sampling. I do not have access to the test samples.

- Website:
  - https://www.kaggle.com/c/plant-pathology-2021-fgvc8
  - https://sites.google.com/view/fgvc8/competitions/plantpathologychallenge2021

## Usage
This dataset is serving as a canonical example for multi-label image classificatino datasets with `timm`. The additions to train & val scripts for this are a WIP...

## Citation
```
Thapa, Ranjita, Zhang, Kai, Snavely, Noah, Belongie, Serge, and Khan, Awais. Plant Pathology 2021 - FGVC8.
https://kaggle.com/competitions/plant-pathology-2021-fgvc8, 2021. Kaggle.
```