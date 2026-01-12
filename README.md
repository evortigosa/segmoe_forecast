# Seg-MoE (Segment-wise Mixture-of-Experts)

This repo is the official implementation for the paper: [Seg-MoE: Multi-Resolution Segment-wise Mixture-of-Experts for Time Series Forecasting Transformers](https://arxiv.org/abs/0000.00000).

## Introduction
We introduce Seg-MoE, a sparse MoE design that routes and processes contiguous time-step segments rather than making independent expert decisions. Token segments allow each expert to model intra-segment interactions directly, naturally aligning with inherent temporal patterns. We integrate Seg-MoE layers into a time-series Transformer and evaluate it on multiple multivariate long-term forecasting benchmarks. Seg-MoE consistently achieves state-of-the-art forecasting accuracy across almost all prediction horizons, outperforming both dense Transformers and prior token-wise MoE models.

## Overall Architecture
Mixture-of-Experts (MoE) designs for sparse conditional computation in Transformer blocks. (a) Standard token-wise MoE: a router computes token-to-expert affinities and selects Top-K routed experts from N experts; the layer output is the weighted sum of the selected expert outputs. (b) Seg-MoE: routing is performed at the segment level, and the output combines Top-K routed experts with an always-active shared expert, providing a stable, dense pathway while preserving sparsity in the routed experts.

<p align="center">
<img src=".\figures\segmoe_architecture.png" width="900" height="" alt="" align=center />
</p>

## TODO List
- A caching mechanism
- Pre-training on large-scale heterogeneous time series datasets

## Usage
1. Install Python 3.12+, and then install the dependencies:

```
pip install -r requirements.txt
```

2. We provide Jupyter notebooks with usage examples in the folder "./notebooks/". You can obtain all multivariate datasets from [[Google Drive]](https://drive.google.com/drive/folders/1MZAg3pELoyvsbW5iHq1-L-4Sn7xrPrLh?usp=sharing), and we also provide methods to download them automatically.

3. Train and evaluate a model.

4. You can reproduce the experiment results by downloading our checkpoints from [[Google Drive]](https://drive.google.com/drive/folders/1Bzifq3w82LoO-edyuu6ThqxBTr5WqrdN?usp=sharing).

## Main Results
We evaluate Seg-MoE on long-term multivariate forecasting benchmarks. Comprehensive forecasting results demonstrate that Seg-MoE effectively enhances the prediction of long-term time series.

### Full-shot Forecasting

<p align="center">
<img src=".\figures\results.png" width="900" height="" alt="" align=center />
</p>

## Citation
If you find this repo helpful, please cite our paper.

```
@article{xxxx,
  title={Seg-MoE: Multi-Resolution Segment-wise Mixture-of-Experts for Time Series Forecasting Transformers},
  author={Evandro S. Ortigossa, ..., Eran Segal},
  journal={xxxx},
  year={xxxx}
}
```

## Acknowledgement
We appreciate the following GitHub repos for their valuable efforts:

Stationary (https://github.com/thuml/Nonstationary_Transformers)

TimeXer (https://github.com/thuml/TimeXer)

Time-MoE (https://github.com/Time-MoE/Time-MoE)

PatchTST (https://github.com/yuqinie98/PatchTST)

GShard (https://github.com/lucidrains/mixture-of-experts)

Switch Transformers (https://github.com/tensorflow/mesh/tree/master/mesh_tensorflow/transformer)

## Contact
Please let us know if you have any suggestions or find out a mistake: 
evandro.scudeleti-ortigossa@weizmann.ac.il or eran.segal@weizmann.ac.il or submit an issue.

## License
This project is licensed under the Apache-2.0 License.