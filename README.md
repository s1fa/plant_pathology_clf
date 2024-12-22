## 1. the task:

ref: https://huggingface.co/datasets/timm/plant-pathology-2021

### Description

#### Problem Statement

Apples are one of the most important temperate fruit crops in the world. Foliar (leaf) diseases pose a major threat to the overall productivity and quality of apple orchards. The current process for disease diagnosis in apple orchards is based on manual scouting by humans, which is time-consuming and expensive.

Although computer vision-based models have shown promise for plant disease identification, there are some limitations that need to be addressed. Large variations in visual symptoms of a single disease across different apple cultivars, or new varieties that originated under cultivation, are major challenges for computer vision-based disease identification. These variations arise from differences in natural and image capturing environments, for example, leaf color and leaf morphology, the age of infected tissues, non-uniform image background, and different light illumination during imaging etc.

Plant Pathology 2020-FGVC7 challenge competition had a pilot dataset of 3,651 RGB images of foliar disease of apples. For Plant Pathology 2021-FGVC8, we have significantly increased the number of foliar disease images and added additional disease categories. This year’s dataset contains approximately 23,000 high-quality RGB images of apple foliar diseases, including a large expert-annotated disease dataset. This dataset reflects real field scenarios by representing non-homogeneous backgrounds of leaf images taken at different maturity stages and at different times of day under different focal camera settings.

#### Specific Objectives

The main objective of the competition is to develop machine learning-based models to accurately classify a given leaf image from the test dataset to a particular disease category, and to identify an individual disease from multiple disease symptoms on a single leaf image.

### Resources

Details and background information on the dataset and Kaggle competition ‘Plant Pathology 2020 Challenge’ were published as a peer-reviewed research article. If you use the dataset for your project, please cite the following [Thapa, Ranjita; Zhang, Kai; Snavely, Noah; Belongie, Serge; Khan, Awais. The Plant Pathology Challenge 2020 data set to classify foliar disease of apples. Applications in Plant Sciences, 8 (9), 2020.](https://bsapubs.onlinelibrary.wiley.com/doi/10.1002/aps3.11390)



## 2. Download the dataset

Dataset repo name: **timm/plant-pathology-2021** (16.1 GB; 18,632 rows)

使用HuggingFace🤗的[镜像网站](https://hf-mirror.com/)

```shell
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download timm/plant-pathology-2021 --repo-type dataset --local-dir pp16_8k
```

## 3. Load and Inspect the dataset

```python
import datasets
ds = datasets.load_from_disk("Path/to/save")

from datasets import load_dataset
dataset = load_dataset("parquet", data_files={'train': 'train.parquet', 'test': 'test.parquet'})

from torchvision.transforms import Compose, ColorJitter, ToTensor

jitter = Compose(
    [ColorJitter(brightness=0.5, hue=0.5), ToTensor()]
)

def transforms(examples):
    examples["pixel_values"] = [jitter(image.convert("RGB")) for image in examples["image"]]
    return examples

dataset = dataset.with_transform(transforms)
```

RMSNorm:

<img src="https://s2.loli.net/2024/12/22/QX27GKWAzcb8mvs.png" alt="image-20241219232253533" style="zoom:50%;" />



## Improvement

1. change to Binary cross entropy
2. turn up the learning rate 0.003 -> 0.03



## 4. Inference / predicition

Use the `inference.py`



### GUI app

Unfinished !
