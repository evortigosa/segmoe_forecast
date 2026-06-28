# Seg-MoE (Segment-wise Mixture-of-Experts)

This repo is the official implementation for the paper: [Seg-MoE: Multi-Resolution Segment-wise Mixture-of-Experts for Time Series Forecasting Transformers](https://arxiv.org/abs/2601.21641).

## Introduction
We introduce Seg-MoE, a sparse Mixture-of-Experts (MoE) design for time-series forecasting that routes contiguous time-step segments as units instead of routing each token independently. Existing MoE forecasters inherit token-wise routing from language models, where consecutive observations that jointly encode a local trend or seasonal event can be split across different experts, preventing any single expert from handling that pattern coherently. Seg-MoE instead operates temporally coherent conditional computation by routing contiguous segments, so temporally adjacent patches are sent to the same expert. We integrate Seg-MoE layers into a time-series Transformer and evaluate it on multiple multivariate long-term forecasting benchmarks. Seg-MoE consistently achieves state-of-the-art forecasting accuracy across almost all prediction horizons, outperforming both dense Transformers and prior token-wise MoE models.

## Overall Architecture
Mixture-of-Experts (MoE) layers introduce sparse conditional computation into Transformer blocks by replacing the dense feed-forward network with a set of expert networks, only a few of which are activated per input. (a) Standard token-wise MoE: a router computes token-to-expert affinities and selects Top-K routed experts from N experts; the layer output is the weighted sum of the selected expert outputs. (b) Seg-MoE: routing is performed at the segment level, so every patch in a segment is handled by the same Top-K experts. The layer output combines the selected routed experts with an always-active shared expert, providing a stable pathway while preserving sparsity in the routed experts.

<p align="center">
<img src=".\figures\segmoe_architecture.png" width="900" height="" alt="" align=center />
</p>

## TODO List
- A caching mechanism
- Pre-training on large-scale heterogeneous time series datasets

## Usage
1. Install Python 3.10+, and then install the dependencies:

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
@article{ortigossa2026seg,
  title={{Seg-MoE}: Multi-Resolution Segment-wise Mixture-of-Experts for Time Series Forecasting Transformers},
  author={Ortigossa, Evandro S. and Segal, Eran},
  journal={arXiv preprint arXiv:2601.21641},
  year={2026}
}
```

## Acknowledgement
We appreciate the following GitHub repos for their valuable efforts:

Time-MoE (https://github.com/Time-MoE/Time-MoE)

PatchTST (https://github.com/yuqinie98/PatchTST)

GShard (https://github.com/lucidrains/mixture-of-experts)

Switch Transformers (https://github.com/tensorflow/mesh/tree/master/mesh_tensorflow/transformer)

Stationary (https://github.com/thuml/Nonstationary_Transformers)

TimeXer (https://github.com/thuml/TimeXer)

## Contact
Please let us know if you have any suggestions or find out a mistake: 
evandro.scudeleti-ortigossa@weizmann.ac.il or eran.segal@weizmann.ac.il or submit an issue.

## License
This project is licensed under the Apache-2.0 License.