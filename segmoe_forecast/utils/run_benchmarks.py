# -*- coding: utf-8 -*-
"""
Segment-wise MoE Forecasting Benchmark using ETT, Weather, ECL, and Traffic datasets
"""

import argparse
import ast
import inspect
import torch
import torch.nn as nn

from dataclasses import fields, replace
from segmoe_forecast.model import TSFTransformer
from segmoe_forecast.model.Config import TinyConfig, SmallConfig, BaseConfig
from segmoe_forecast.data_provider.loaders import get_ett_data_loaders, get_custom_data_loaders
from segmoe_forecast.utils import CosineLRDecay, EarlyStopping, LoadBalancingLoss, Trainer
from segmoe_forecast.utils.Metrics import eval_forecast_horizons



CONFIG_MAP= {
    "base": BaseConfig, "tiny": TinyConfig, "small": SmallConfig,
}


def parse_value(value:str):
    """
    Convert CLI string to Python object when possible.
    Examples:
      '256'     -> 256
      '0.1'     -> 0.1
      'True'    -> True
      'False'   -> False
      'None'    -> None
      '[1,2,3]' -> [1, 2, 3]
      'gelu'    -> 'gelu'
    """
    low= value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "none":
        return None

    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def parse_overrides(items):
    overrides= {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE.")
        key, value= item.split("=", 1)
        key= key.strip()
        value= parse_value(value.strip())
        overrides[key]= value

    return overrides


def build_parser():
    parser= argparse.ArgumentParser(description="Segment-wise MoE forecasting train-test script")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable or disable text infos")
    # model
    parser.add_argument("--model-size",  type=parse_value, default='small', help="Type of model configuration")
    parser.add_argument("--block-size",  type=int, default=512, help="Input sequence length / context window")
    parser.add_argument("--patch-width", type=int, default=8, help="Patch width")
    parser.add_argument("--width-factor",type=float, default=4, help="Output patch width")
    parser.add_argument("--n-outputs",   type=int, default=96, help="Prediction horizon / number of outputs")
    parser.add_argument("--channels",    type=int, default=7, help="Number of input channels")
    parser.add_argument(
        "--exp-segment-size", type=parse_value, default="[3,5,5,5]",
        help="Number of contiguous tokens handled by each Seg-MoE layer. Can be a single int or a list of ints"
    )
    parser.add_argument(
        "--set", dest="model_overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override any model config field, e.g. --set d_model=256 --set dropout=0.1"
    )
    # data
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--root-path",  type=str, default='./', help="Path of the local CSV file")
    parser.add_argument("--dataset-name", type=str, default='ETTh1', help="Name of the benchmark data")
    parser.add_argument("--from-csv", action=argparse.BooleanOptionalAction, default=True,
                        help="False: read data from NeuralForecast. True: provide a local CSV file")
    # train
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--checkpoint-dir", type=str, default='checkpoints', help="Path to the checkpoint repo")
    parser.add_argument("--checkpoint-file",type=str, default='tsft_checkpoint', help="File name of the checkpoint")
    parser.add_argument("--plot-file", type=str, default='tsft_training', help="File name for training plots")
    parser.add_argument("--max-lr", type=float, default=3.2e-4, help="Max learning rate")
    parser.add_argument("--min-lr", type=float, default=1.2e-5, help="Min learning rate")
    parser.add_argument("--warmup-portion",type=float, default=0.1, help="Percentage of steps as warmup")
    parser.add_argument("--weight-decay",  type=float, default=1e-1, help="AdamW weight_decay")
    parser.add_argument("--setup-opt", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable or disable model setup_optimizer on weight decayed parameters")
    parser.add_argument("--loss", type=str, default='huber', help="Loss criterion can be HuberLoss or MSELoss")
    parser.add_argument("--stop-patience", type=int, default=5, help="Number of patience epochs for early stopping")
    parser.add_argument("--stop-min",   type=float, default=1e-6, help="Min delta for early stopping")
    parser.add_argument("--clip-grad", type=parse_value, default=None,
                        help="Set a value (float) to clip_grad_norm_ on training")
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable or disable model training")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable or disable bf16")
    parser.add_argument("--moe-metrics", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable or disable MoE tracking metrics")
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable or disable model test")
    parser.add_argument("--show-tqdm", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable or disable tqdm status bar on training/test")
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable or disable saving training/validation plots")
    parser.add_argument("--plot-cut-first", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable or disable first epoch results in plot files")
    parser.add_argument("--seed", type=parse_value, default=None, help="Random number generator seed")

    return parser


def count_parameters(model) -> None:
    total_params= sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of model parameters: {total_params:,}')


def setup_model_from_checkpoint(filename, checkpoint_dir, verbose=True):
    trainer= Trainer(
        model=None, device="cpu", train_loader=None, train_ds_scaler=None, val_loader=None, test_loader=None,
        criterion=None, optimizer=None, checkpoint_dir=checkpoint_dir, filename=filename, verbose=verbose
    )
    model= trainer.build_model(filename=None, checkpoint_dir=checkpoint_dir)
    del trainer

    return model, model.config


def setup_model(model_size, args):
    if model_size.lower() not in ('tiny', 'small', 'base'):
        raise ValueError("model_size must be one of: 'tiny', 'small', 'base'.")

    config_cls= CONFIG_MAP[model_size.lower()]
    config= config_cls()
    cli_overrides= parse_overrides(args.model_overrides)

    # explicit arguments that should always override preset defaults
    explicit_overrides= {
        "patch_width": args.patch_width,
        "channels": args.channels,
        "n_outputs": args.n_outputs,
        "width_factor": args.width_factor,
        "block_size": args.block_size,
        "exp_segment_size": args.exp_segment_size,
    }
    # generic overrides take final precedence
    overrides= {**explicit_overrides, **cli_overrides}

    # validate override names
    valid_fields= {f.name for f in fields(config)}
    unknown= set(overrides) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config field(s) for {config_cls.__name__}: {sorted(unknown)}")
    # apply overrides
    config= replace(config, **overrides)

    model= TSFTransformer.from_config(config)

    return model, model.config


def setup_data_loaders(
    btc_size, root_path, dataset_name, from_csv, patch_width, block_size, width_factor
):
    time_covariates= False

    if dataset_name.lower() in {'ettm1','ettm2','etth1','etth2'}:
        root_path= 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/'  # official ETT data repo
        from_csv= True

        if dataset_name.lower() == 'ettm1':
            dataset_name_1= 'ETTm1'
            dataset_name_2= 'ETTm2'
        elif dataset_name.lower() == 'ettm2':
            dataset_name_1= 'ETTm2'
            dataset_name_2= 'ETTm1'
        elif dataset_name.lower() == 'etth1':
            dataset_name_1= 'ETTh1'
            dataset_name_2= 'ETTh2'
        else:  #name == 'etth2'
            dataset_name_1= 'ETTh2'
            dataset_name_2= 'ETTh1'

        (
            train_loader, val_loader, _, _, tds_scaler, _,
            test_loader_96, test_loader_192, test_loader_336, test_loader_720, _, _, _, _,

        )= get_ett_data_loaders(
            root_path, dataset_name_1, dataset_name_2, from_csv, btc_size, time_covariates, patch_width,
            block_size, width_factor
        )
    else:
        (
            train_loader, val_loader, tds_scaler,
            test_loader_96, test_loader_192, test_loader_336, test_loader_720,

        )= get_custom_data_loaders(
            root_path, dataset_name, from_csv, btc_size, time_covariates, patch_width,
            block_size, width_factor
        )

    return (
        train_loader, val_loader, tds_scaler,
        test_loader_96, test_loader_192, test_loader_336, test_loader_720,
    )


def setup_trainer(
    model, device, use_fused, train_loader, val_loader, test_loader, scaler_obj,
    checkpoint_dir, filename, epochs=10, max_lr=3.2e-3, min_lr=1.2e-4, warmup_portion=0.1, weight_decay=1e-1,
    setup_optimizer=False, loss='huber', stop_patience=5, stop_min_delta=1e-6, verbose=True, disable_tqdm=True,
):
    config= model.config
    steps = len(train_loader) * epochs
    warmup_steps= steps * warmup_portion
    max_steps= steps

    if setup_optimizer:
        optimizer= model.setup_optimizer(
            learning_rate=max_lr, weight_decay=weight_decay, betas=(0.9, 0.95), verbose=verbose
        )
    else:
        optimizer= torch.optim.AdamW(
            model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=weight_decay,
            eps=1e-10, fused=use_fused
        )
    # for decreasing learning rate -- the CosineLRDecay is designed to be used per step
    scheduler= CosineLRDecay(optimizer, min_lr, max_lr, warmup_steps, max_steps)
    # terminate training when the validation loss (per epoch) does not improve
    early_stopping= EarlyStopping(patience=stop_patience, min_delta=stop_min_delta)

    if loss.lower() == 'huber':
        # See https://arxiv.org/abs/2409.16040
        criterion= nn.HuberLoss(reduction='none', delta=2.0)
    else:
        criterion= nn.MSELoss(reduction='none')
    aux_criterion= LoadBalancingLoss(config.n_experts, config.top_k_experts, alpha=0.02)

    trainer_obj= Trainer(
        model, device, train_loader, scaler_obj, val_loader, test_loader, criterion, optimizer,
        scheduler, aux_criterion, early_stopping, use_time_features=False, do_validation=True,
        checkpointing=True, checkpoint_dir=checkpoint_dir, filename=filename, verbose=verbose,
        disable_tqdm=disable_tqdm,
    )

    return trainer_obj


def main():
    args= build_parser().parse_args()
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    use_fused= False
    use_flashattn= False

    if device== 'cuda':
        # TF32 computationally more efficient (slightly the same precision of FP32)
        torch.set_float32_matmul_precision('high')
        # torch.backends.cudnn.fp32_precision= 'tf32'
        # enable flash attention
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cudnn.deterministic= True
        # create AdamW optimizer and use the fused version of it is available
        fused_available= 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # fused is a lot faster when it is available and when running on cuda
        use_fused= fused_available
        use_flashattn= torch.backends.cuda.flash_sdp_enabled()

    if args.seed is not None:
        torch.manual_seed(int(args.seed))

    verbose= args.verbose
    if verbose:
        print(f"[INFO] Device: {device}; Using fused AdamW: {use_fused}; FlashAttention available: {use_flashattn}")

    model_size= args.model_size
    check_dir = args.checkpoint_dir
    check_file= args.checkpoint_file
    plot_file = args.plot_file

    if model_size is None:  # when None, try to build a model from the checkpoint
        ts_model, model_config= setup_model_from_checkpoint(check_file, check_dir, verbose)
    else:
        ts_model, model_config= setup_model(model_size, args)

    ts_model= ts_model.to(device)
    if verbose:
        count_parameters(ts_model)

    btc_size= args.batch_size
    root_path= args.root_path
    dataset_name= args.dataset_name
    from_csv= args.from_csv

    (
        enc_train_loader, enc_val_loader, enc_tds_scaler,
        test_loader_96, test_loader_192, test_loader_336, test_loader_720,

    )= setup_data_loaders(
        btc_size, root_path, dataset_name, from_csv, model_config.patch_width, model_config.block_size,
        model_config.width_factor
    )
    if verbose:
        print(f"[INFO] {dataset_name} data -- number of batches (train, val, test-96): {len(enc_train_loader)}, "
              f"{len(enc_val_loader)}, {len(test_loader_96)}")

    epochs= args.epochs
    disable_tqdm= not args.show_tqdm

    trainer= setup_trainer(
        ts_model, device, use_fused, enc_train_loader, enc_val_loader, test_loader_96, enc_tds_scaler,
        check_dir, check_file, epochs, args.max_lr, args.min_lr, args.warmup_portion, args.weight_decay, args.setup_opt,
        args.loss, args.stop_patience, args.stop_min, verbose, disable_tqdm
    )

    if args.train:
        trainer.train(epochs, use_bf16=args.bf16, clip_grad=args.clip_grad, get_moe_metrics=args.moe_metrics)

        if args.save_plots:
            trainer.plot_results(cut_first_epoch=args.plot_cut_first, show_plot=False, save_charts=True, file_name=f"{plot_file}_losses")
            trainer.plot_expert_routing_diagnostics(show_plot=False, save_charts=True, file_name=f"{plot_file}_expert_routing")
            if model_config.n_experts <= 8:
                trainer.plot_expert_usage_global(show_plot=False, save_charts=True, file_name=f"{plot_file}_expert_usage_global")
                trainer.plot_expert_usage_layerwise(show_plot=False, save_charts=True, file_name=f"{plot_file}_expert_usage_layer")
            else:
                trainer.plot_expert_usage_global_heatmap(show_plot=False, save_charts=True, file_name=f"{plot_file}_expert_usage_heatmap")
                trainer.plot_expert_usage_layerwise_heatmap(show_plot=False, save_charts=True, file_name=f"{plot_file}_expert_usage_heatmap_layer")

    if args.test:
        _, _= trainer.load_checkpoint(filename=None, checkpoint_dir=check_dir)

        avg_mse, avg_mae= eval_forecast_horizons(
            trainer, dataset_name, test_loader_96, test_loader_192, test_loader_336, test_loader_720
        )
        if verbose:
            print(f"\nAverage MSE: {avg_mse:.4f}, Average MAE: {avg_mae:.4f}")


if __name__ == '__main__':
    main()
