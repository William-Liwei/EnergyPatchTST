# EnergyPatchTST: Multi-scale Time Series Transformers with Uncertainty Estimation for Energy Forecasting

This is an official implementation of  [EnergyPatchTST: Multi-scale Time Series Transformers with Uncertainty Estimation for Energy Forecasting](https://link.springer.com/chapter/10.1007/978-981-96-9815-8_27).

![license](https://img.shields.io/badge/License-Apache_2.0-blue.svg) [![Issues](https://badgen.net/github/issues/William-Liwei/EnergyPatchTST?color=red)](https://github.com/William-Liwei/EnergyPatchTST/issues) [![PRs](https://badgen.net/github/prs/William-Liwei/EnergyPatchTST?color=yellow)](https://github.com/William-Liwei/EnergyPatchTST/pulls) ![GitHub repo size](https://img.shields.io/github/repo-size/william-liwei/energypatchtst?color=green) ![GitHub Repo stars](https://badgen.net/github/stars/William-Liwei/EnergyPatchTST?color=orange) ![GitHub forks](https://badgen.net/github/forks/William-Liwei/EnergyPatchTST?color=pink)

## Authors

- Wei Li, Shanghai University;
- Zixin Wang, Shanghai University;
- Qizheng Sun, Shanghai University;
- Qixiang Gao, Fudan University;
- *Fenglei Yang, Shanghai Univeristy;

## Introduction & Motivation

Accurate energy time series prediction is crucial for power generation planning and allocation. However, existing deep learning methods face limitations due to the multi-scale time dynamics and irregularity of real-world energy data. Our proposed model, EnergyPatchTST, is an extension of the Patch Time Series Transformer specifically designed for energy forecasting. It addresses the unique challenges of energy time series by incorporating multi-scale feature extraction, uncertainty estimation through a probabilistic prediction framework, integration of future known variables, and a pre-training and fine-tuning strategy leveraging transfer learning.

## Prerequisites

Ensure you are using Python 3.9 and install the necessary dependencies by running:

```
pip install -r requirements.txt
```

## Data set preparation

Due to the storage space limitations of GitHub, some data sets may need to be downloaded separately. However, some small data sets have been included in the repository for demonstration purposes. If you encounter any issues with the data sets, please contact us via email [liwei008009@163.com](mailto:liwei008009@163.com).

## Model Overview

EnergyPatchTST is built upon the PatchTST architecture and introduces several key enhancements for energy forecasting. It processes time series data at different temporal resolutions to capture short-term fluctuations and long-term trends. The model also incorporates future known variables such as weather forecasts through a specialized projection pathway. Additionally, it provides reliable uncertainty estimates using a Monte Carlo dropout mechanism and employs a pre-training and fine-tuning approach to effectively leverage general time series datasets.

## Key Features

- **Multi-scale Feature Extraction**: Captures patterns at different time resolutions, from immediate fluctuations to daily and seasonal patterns.
- **Uncertainty Estimation**: Provides probabilistic forecasts with calibrated prediction intervals using a Monte Carlo dropout mechanism.
- **Future Variables Integration**: Incorporates known future variables such as temperature and wind speed forecasts to improve forecast accuracy.
- **Pre-training and Fine-tuning**: Utilizes transfer learning by pre-training on general time series datasets and fine-tuning on specific energy datasets.

## Experiment Results

The experimental results demonstrate that EnergyPatchTST consistently outperforms baseline methods across different prediction horizons. It achieves a reduction in forecasting error by 7-12% while providing reliable uncertainty estimates. The model shows significant performance improvements for longer horizons, highlighting the effectiveness of its multi-scale approach for capturing long-term patterns.

## Commercial Use

This repository contains a partial implementation of the EnergyPatchTST model, which is allowed for commercial use. However, if you are interested in using the complete implementation, please contact us via email to negotiate business cooperation.

## Contact

If you have any questions or need further assistance, please contact [liwei008009@163.com](mailto:liwei008009@163.com) or submit an issue in the repository.

## Acknowledgement

We extend our gratitude to the following repositories for their valuable code and datasets:

- https://github.com/yuqinie98/PatchTST
- https://github.com/thuml/Time-Series-Library
- https://github.com/thuml/Autoformer
- https://github.com/Hank0626/WFTNet
- https://github.com/William-Liwei/FreqHybrid
- https://github.com/William-Liwei/SWIFT
- https://github.com/William-Liwei/ScatterFusion
- https://github.com/William-Liwei/TimeFlowDiffuser
- https://github.com/William-Liwei/LWSpace

## Citation

If you find this repo useful, please cite it as follows:

```
@InProceedings{10.1007/978-981-96-9815-8_27,
author="Li, Wei
and Wang, Zixin
and Sun, Qizheng
and Gao, Qixiang
and Yang, Fenglei",
editor="Huang, De-Shuang
and Pan, Yijie
and Chen, Wei
and Li, Bo",
title="EnergyPatchTST: Multi-scale Time Series Transformers with Uncertainty Estimation for Energy Forecasting",
booktitle="Advanced Intelligent Computing Technology and Applications",
year="2025",
publisher="Springer Nature Singapore",
address="Singapore",
pages="319--330",
abstract="Accurate and reliable energy time series prediction is of great significance for power generation planning and allocation. At present, deep learning time series prediction has become the mainstream method. However, the multi-scale time dynamics and the irregularity of real data lead to the limitations of the existing methods. Therefore, we propose EnergyPatchTST, which is an extension of the Patch Time Series Transformer specially designed for energy forecasting. The main innovations of our method are as follows: (1) multi-scale feature extraction mechanism to capture patterns with different time resolutions; (2) probability prediction framework to estimate uncertainty through Monte Carlo elimination; (3) integration path of future known variables (such as temperature and wind conditions); And (4) Pre-training and Fine-tuning examples to enhance the performance of limited energy data sets. A series of experiments on common energy data sets show that EnergyPatchTST is superior to other commonly used methods, the prediction error is reduced by 7--12{\%}, and reliable uncertainty estimation is provided, which provides an important reference for time series prediction in the energy field.",
isbn="978-981-96-9815-8"
}
```

## Star History

<a href="https://www.star-history.com/#William-Liwei/EnergyPatchTST&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=William-Liwei/EnergyPatchTST&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=William-Liwei/EnergyPatchTST&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=William-Liwei/EnergyPatchTST&type=Date" />
 </picture>
</a>
