# Large Language Models Understand Layout

This repository contains the methods and results described in the paper titled "Large Language Models Understand
Layout".

## Introduction

Large language models (LLMs) demonstrate extraordinary abilities in a wide range of natural language processing tasks.
In this paper, we show that, beyond text understanding capability, LLMs are capable of processing text layout that is
denoted by spatial markers. They are able to answer questions that require explicit spatial perceiving and reasoning,
while a drastic performance drop is observed when the spatial markers from the original data are excluded. We perform a
series of experiments with the GPT-3.5, Baichuan2, Llama2 and ChatGLM3 models on various types of layout sensitive
datasets for further analysis. The experimental results reveal that the layout understanding ability of LLMs is mainly
introduced by the coding data for pretraining, which is further enhanced at the instruction-tuning stage. In addition,
layout understanding can be enhanced by integrating low-cost, auto-generated data produced by a novel text game.
Finally, we show that layout understanding ability is beneficial for building efficient visual question answering
systems.

## System Requirements

### Hardware requirements

This project requires a GPU server with at least 32GB GPU memory. All the
results present in the paper were obtained using Nvidia 8*A100 server (80 GB version).

### Software requirements

#### OS Requirements

This project has been tested on the following systems:

- Linux: Ubuntu 18.04

#### Python Dependencies

Install packages listed in requirements.txt. Python version has been tested is 3.8.

```
pip install -r requirements.txt
```

## Datasets

The datasets can be downloaded from Alipan (https://www.alipan.com/s/v6FrwXjq166) or Google Drive (https://drive.google.com/drive/folders/1w8LD-gfskhn_rK8dxO5WZ4LPqRY4HPDQ?usp=sharing).

## Evaluate

Evaluate the performance of different LLMs on the datasets. For example, to evaluate the
dataset `TextLayoutQA`, run the following script:

```
cd dataset/TextLayoutQA
python evaluate.py
```

## Instruction-tuning

Script for instruction-tuning is shown below, all the settings can be referred to the configuration files
under `train_args` folder.

```
torchrun --nproc_per_node=8 train_qlora.py --train_args_file train_args/baichuan2-7b-sft-qlora.json
```

## Test
Script for testing the instruction-tuning performance is shown below.

```
cd Firefly
python eval.py
```


