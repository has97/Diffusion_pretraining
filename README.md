# Code for : [Enhancing the Domain Robustness of Self-Supervised Pre-Training with Synthetic Images]
[![Conference](https://img.shields.io/badge/ICASSP-2024-4b44ce)](https://2024.ieeeicassp.org/)

<a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/-Python_3.7-blue?logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch_1.10-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning_2.1.2-792ee5?logo=pytorchlightning&logoColor=white"></a>
## Overview

This repository contains the code for Enhancing the Domain Robustness of Self-Supervised Pre-Training with Synthetic Images. We present a novel method for improving the adaptability of self-supervised (SSL) pre-trained models across different domains.

**Note**: This code is built on top of the work of [Solo-learn](https://github.com/vturrisi/solo-learn). Please refer to the original repository/paper for more details on the core algorithms and models.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)


## Prerequisites

The requirements of this repo are requirements of [Solo-learn](https://github.com/vturrisi/solo-learn) and diffusers library for InstructPix2Pix model

## Usage 

Modify the paths in the imagenet\_aug.py and generate\_augmentation.py according to the location of the datasets and folders. Then run the following command to generate the diffused images.

```bash
python test.py
```

After the images are generated. Update the paths to multi-domain images in the pretrain\_dataloader.py. This should correpond to the path where the diffused images are stored. The pre\_training and linear\_evaluation commands are same as the [Solo-learn](https://github.com/vturrisi/solo-learn) repo with appropriate changes in the config files. For training with diffused images, format in config file must be changed to diffused and for Imagenet-100 training choose format to either image\_folder or dali.

