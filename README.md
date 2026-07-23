# Meta-DAN: towards an efficient prediction strategy for page-level handwritten text recognition

**<span style="color: #A6351B;">Warning: this project is under a research/academic usage only license, preventing any usage for commercial purposes without permission.</span>**

This repository is the official pytorch implementation of the paper:  
Denis Coquenet, "Meta-DAN: towards an efficient prediction strategy for page-level handwritten text recognition", Pattern Recognition, 2026.

The paper is available [here](https://doi.org/10.1016/j.patcog.2026.113373).


Table of contents:
1. [Getting Started](#Getting-Started)
2. [Quick prediction](#Make-a-prediction)
3. [Datasets](#Datasets)
4. [Training](#Train-a-model-by-yourself)
5. [Citation](#Citation)
5. [License](#License)


More works (papers, models, datasets, demos) are available on my [academic page](https://people.irisa.fr/Denis.Coquenet/publications.html).

# Getting started

Experiments were performed with Python 3.10.14, Pytorch 2.4 and CUDA 12.1.

## Installation

```
conda create -n metadan python==3.10
conda activate metadan

git clone git@github.com:FactoDeepLearning/META-DAN.git
cd dar

pip install -e .
```

# Available models

This repository contains several page-level recognition strategies :

| Name | model_name (for cli) | Associated paper |
|------|----------------------|------------------|
| DAN  | dan                  | TPAMI 2023 ([Arxiv](https://arxiv.org/abs/2203.12273))|
| Faster DAN  | faster_dan    | ICDAR 2023 ([Arxiv](https://arxiv.org/abs/2301.10593))|
| W-DAN  | w_dan              | PR 2026 ([Arxiv](https://arxiv.org/abs/2504.03349))|
| MT-DAN  | mt_dan            | PR 2026 ([Arxiv](https://arxiv.org/abs/2504.03349))|
| Meta-DAN  | meta_dan        | PR 2026 ([Arxiv](https://arxiv.org/abs/2504.03349))|

Also, the 'line_ctc' model is also available for the line-level pre-training stage.

# Download pretrained model weights

Weights are available on [HuggingFace](https://huggingface.co/FactoDeepLearning/MetaDAN) and one can refer to them directly in cli commands with automatic downloading process. 

Checkpoint names are defined as follows: {approach_name}\_{encoder_name}\_{dataset}.

with: 
- {approach_name} in [line_ctc, dan, meta_dan, faster_dan, w_dan, mt_dan].
- {encoder_name} in [fcn_m, convnext-v2-b, convnext-v2-n, convext-v2-t]
- {dataset_name} in [bressay, casia2, eparchos, esposalles, iam, latin, maurdor_c3, maurdor_c4, read2016, rimes2009, rimes2024, scribblelens]

See [HuggingFace](https://huggingface.co/FactoDeepLearning/MetaDAN/tree/main) for the exact available combinations.

# Make a prediction

If you want to use a pretrained model and perform inference on your own document image, you can use:
```
dar model --mode=predict --network={model_name} --checkpoint={path_to_checkpoint} --pSource={path_to_image_or_fold} --pTarget={path_to_store_prediction}
```

For example, performing prediction from a pre-trained line-level model:
```
dar model --mode=predict --network= --checkpoint=line_fcn_m_iam --pSource=Datasets/formatted/iam_line/train/train_a01-000u_0.jpg --pTarget=predictions/line_iam/
```

# Datasets

## Use an existing dataset

Here are the datasets with formatting scripts and URL to download them:

| Dataset  | Link          | | Dataset  | Link          |
| :--------------- |:---------------:| :---------------:|:---------------:|:---------------:|
| BRESSAY  | [Download](https://tc11.cvc.uab.es/datasets/BRESSAY_1) | |  MAURDOR  | [Download](https://catalogue.elra.info/en-us/repository/browse/ELRA-E0045/)          | |READ2016  | [Download](https://zenodo.org/records/1164045#.YiINkBvjKEA)          |
| CASIA  | [Download](https://nlpr.ia.ac.cn/databases/handwriting/Download.html)            | | READ2016 | [Download](https://zenodo.org/records/1164045#.YiINkBvjKEA) |
| EPARCHOS  | [Download](https://zenodo.org/records/4095301)          |  | RIMES2024  | [Download](https://zenodo.org/records/10812725)          |  
| IAM  | [Download](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)          | | ScribbleLens  | [Download](https://openslr.org/84/)          | 

To use one of them, you must:
1 - Download it
2 - Use *dar format* command line to make it compatible with the package, as follows:

```
dar format {dataset_name} --data-folder={dataset_folder_source_path} --output-dataset-name={output_fold_name} --output-folder={path_to_output_fold} --level={level}
```

with:
- {dataset_name} in [berssay, casia2, eparchos, iam, maurdor, read2016, rimes2009, rimes2024, scribblelens] 
- {level} in [line, paragraph, page]

Example to format IAM dataset stored at *./Datasets/raw/IAM* at line level into *./Datasets/formatted/iam_line*:

```
dar format iam --data-folder=Datasets/raw/IAM --output-dataset-name=iam_line --output-folder=Datasets/formatted/ --level=line
```

### Dataset splits
The dataset splits used in the paper are available on the following folder: dar/dataset_format/splits


## Create your own datasets

If you want to use your own dataset for training, you must format it so that you reach the following fold structure :

dataset_name
├── train
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └─ train_image.png
├── val
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └─ validation_image.png
├── test
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └─ test_image.png
├── charset.pkl
└── labels.json

where charset.pkl corresponds to a list of characters (the vocabulary), saved with pickle,
and labels.json corresponds to a dictionary separating all files into train, val and test split (keys are relative paths from dataset folder, and values contains their ground truth as a list of characters).
{
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"train": [
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"dataset_name/train/train_image.png": {
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"text": ["g", "r", "o", "u", "n","d", " ", "t", "r", "u", "t", "h"]
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}, 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;],
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"val": [...],
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"test": [...],
}


If you only want to test a pretrained model on your document image, please refer to [the prediction section](#Make-a-prediction).
Your formatted datasets must be located at *dar/Datasets/formatted/dataset_name*.

# Train a model by yourself

**Important note:** default configuration files employ synthetic data generation. You must add at least one font file to the *Fonts/* folder to make it work (for example, from https://www.1001fonts.com).

Here are some training use-case examples:

### Line-level pretraining
To perform the pretraining stage (line-level with CTC):
```
dar model --mode=train --network=line_ctc --dataset={dataset_name} --output-folder=line_model_ctc
```

### Page-level training
To perform the main training stage with MT-DAN, transferring weights from the pretraining stage, you could do:
```
dar model --mode=train --network=mt_dan --dataset={dataset_name} --output-folder=mt_dan_model --transfer=outputs/line_model_ctc/checkpoints/ --syn-config=page_syn_mix.yaml
```

And further fine-tuning with META-DAN:
```
dar model --mode=train --network=meta_dan --dataset={dataset_name} --output-folder=meta_dan_model --transfer=outputs/mt_dan_model/checkpoints/ --syn-config=page_syn_mix.yaml
```

### Evaluate the trained model
The model is regularly evaluated on the val set during training phase.
To evaluate it on the test set:
```
dar model --mode=eval --network=meta_dan --dataset={dataset_name} --output-folder=meta_dan_model 
```

### Useful command-line options
Here is a non exhaustive list of useful command-line arguments you may use to custom your training:

|Argument |Type | Description|
|---------|-----|------------|
|**Generic**
|--dataset|str|Dataset folder name on which you want to train/eval (mono-dataset training)|
|--dataset-config|str|File name of configuration file for multi-dataset training|
|--encoder|str| Choice between "fcn", "fcn-m256", "convnext-v2-n", "convnext-v2-t", "convnext-v2-b"
|--output-folder|str|Name of the output folder (for weights and logs)|
|--syn-config|str|Name of config yaml path for synthetic document generation|
|--max-num-samples|int|Maximum number of training samples (training ends when reaching it)|
|--training-time|float|Maximum number of training hours (training ends when reaching it)|
|--save-interval|int|Number of training samples between to evaluation on val set (inference can be long)|
|--eval-start|int|Number of training samples before first evaluation on val set (inference can be even longer at first steps)|
|--all-batch-size|int|Size of the mini-batch used for training/val/test|
|--no-amp|None|Automatic-mixed precision is used by default (disable it)|
|--ddp|None|Enable data distributed parallel (beta)|
|--load-images-on-the-fly|None|Disable loading all images in RAM (necessary for large datasets)|
|--transfer|str|Path of checkpoint to use for transfer learning purpose|
|--lr-all, --lr-encoder, --lr-decoder|float|Learning rate to use for part or whole architecture.|
|**For DAN, MT-DAN, W-DAN and META-DAN**
|--max-char-prediction | int | Maximum number of predictions (decoding stage ends when reaching it)|
|**Using W-DAN**
|--token-win|int| Size of the query window ($w$)|
|**Using MT-DAN**
|--num-token-heads | int | Number of heads = max number of predictions per decoding iteration ($m$)|
|--decoding-mt-mode | str | "static" (preserve all predictions) or "dynamic" (preserve predictions if > threshold)|
|--decoding-threshold | float | Threshold for "dynamic" mode|
|--decoding-num-preds | int | Number of predictions to preserve per decoding iteration, for "static" mode |
|**Using ConvNext V2 encoder**
|--adapt-encoder-downscaling| None|Reduce downsampling factor from (32, 32) down to (32, 8)|
**Using synthetic generation**
|--syn-init-proba|float|Percentage of synthetic data for training at first|
|--syn-end-proba|float|Percentage of synthetic data for training at last|
|--syn-num-samples|int|Number of training samples to linearly go from syn-init-proba to syn-end-proba|
|--curr-num-samples|int|Number of training samples to linearly go from 1 to max number of synthetic lines/page|
|--start-curr-at-sample|int|Start curriculum stage after some training samples|
|--syn-source|str|Text used to generate synthetic samples: "dataset" (default) or "wiki-XX" (wiki-fr,wiki-en,...)|
**Using BPE tokenizer** ||(Character tokenizer is used by default)
|--tokenizer|str|"char" (default) or "bpe" (Byte-Pair Encoding)|
|--tok-model-name|str|Pretrained tokenizer file name to use|
|--tokenizer-source|str|Text on which to train tokenizer: "dataset" or "wiki-XX" (wiki-fr,wiki-en,...)|
|--tok-num-tokens|int|Size of vocabulary|


# Citation
To cite the paper:

```bibtex
@article{Coquenet2026,
    author={ Denis Coquenet },
    journal={Pattern Recognition},
    title={Meta-DAN: towards an efficient prediction strategy for page-level handwritten text recognition},
    year={2026},
    pages={113373},
    volume={117},
    doi={10.1016/j.patcog.2026.113373},
} 
```

To cite the repository, please refer to the CITATION.cff file.


# License
**This project is under a research/academic usage only license, preventing any usage for commercial purposes without permission.**  
It is built upon the DAN source code from the following publication:  
Coquenet, Denis and Chatelain, Clément and Paquet, Thierry. “DAN: a Segmentation-free Document Attention Network for Handwritten Document Recognition” 10.1109/TPAMI.2023.3235826, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023.

If you are interested in using this work for commercial purposes, you must buy a license.

More details will come soon.
