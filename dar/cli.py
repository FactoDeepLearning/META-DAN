import argparse
import os
import copy
from dar.text_recognition.run import run
from dar.generic.config_utils import load_yaml_file, preprocess_params, DEFAULT_CONFIG
from dar.dataset_format import add_format_dataset_parser


def add_model_parser(subcommands):
    parser = subcommands.add_parser("model")
    parser.add_argument(
        "--mode",
        required=True,
        choices=['train', 'eval', 'predict', 'eval_all'],
        help="The action to perform",
    )

    parser.add_argument(
        "--network",
        required=True,
        choices=['line_ctc', 'dan', 'faster_dan', 'w_dan', 'mt_dan', 'meta_dan'],
        help="The network to use ",
    )

    parser.add_argument(
        "--config",
        required=False,
        help="Configuration file.",
    )

    parser.add_argument(
        "--train-batch-size",
        required=False,
        help="Overwrite training batch size in config.",
        type=int
    )

    parser.add_argument(
        "--seed",
        required=False,
        default=0,
        help="Seed to use for randomness and experiment replicability.",
        type=int
    )

    parser.add_argument(
        "--loss-reduction",
        required=False,
        help="Reduction to use for loss (see torch options).",
        type=str
    )

    parser.add_argument(
        "--drop-last-res",
        required=False,
        help="Overwrite drop_last_res in config.",
        action="store_true"
    )

    parser.add_argument(
        "--drop-all-res",
        required=False,
        help="Overwrite drop_all_res in config.",
        action="store_true"
    )

    parser.add_argument(
        "--track-hist-tokens",
        required=False,
        help="Count the number of occurrences of each token at training time.",
        action="store_true"
    )

    parser.add_argument(
        "--add-weights-overlap-penalty",
        required=False,
        help="Penalize loss for several iteration focusing on same part to avoid infinite loops.",
        action="store_true"
    )

    parser.add_argument(
        "--add-loop-teacher-forcing",
        required=False,
        help="Add loop error in teacher forcing when training.",
        action="store_true"
    )

    parser.add_argument(
        "--add-loop-syn",
        required=False,
        help="Add identical text lines in generated synthetic documents.",
        action="store_true"
    )

    parser.add_argument(
        "--weights-penalty-factor",
        default=1,
        help="Factor in loss.",
        type=float
    )

    parser.add_argument(
        "--max-num-epochs",
        required=False,
        help="Overwrite max_nb_epochs in config.",
        type=int
    )

    parser.add_argument(
        "--save-mode",
        required=False,
        choices=["epoch", "sample"],
        help="Overwrite save_weights mode in config.",
    )

    parser.add_argument(
        "--save-interval",
        required=False,
        type=int,
        help="Overwrite save_weights interval in config.",
    )

    parser.add_argument(
        "--eval-mode",
        required=False,
        choices=["epoch", "sample", "step"],
        help="Overwrite eval_on_valid mode in config.",
    )

    parser.add_argument(
        "--eval-start",
        required=False,
        type=int,
        help="Overwrite eval_on_valid start_at in config.",
    )

    parser.add_argument(
        "--eval-interval",
        required=False,
        type=int,
        help="Overwrite eval_on_valid interval in config.",
    )

    parser.add_argument(
        "--eval-weight-key",
        default="best",
        choices=["best", "last"],
        type=str,
        help="Which weight to use for final evaluation (best or last).",
    )

    parser.add_argument(
        "--max-num-samples",
        required=False,
        help="Overwrite max_num_samples in config.",
        type=int
    )

    parser.add_argument(
        "--grad-acc",
        required=False,
        help="Overwrite gradient accumulation in config.",
        type=int
    )

    parser.add_argument(
        "--grad-clip-max",
        required=False,
        help="Clip gradient with max threshold.",
        type=float
    )

    parser.add_argument(
        "--val-batch-size",
        required=False,
        help="Overwrite val batch size in config.",
        type=int
    )

    parser.add_argument(
        "--test-batch-size",
        required=False,
        help="Overwrite test batch size in config.",
        type=int
    )

    parser.add_argument(
        "--all-batch-size",
        required=False,
        help="Overwrite train/val/test batch size in config.",
        type=int
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Do not use automatic mixed precision (overwrite use amp in config).",
    )

    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Use Distributed Data Parallel (for multi-gpu only, overwrite use ddp in config).",
    )

    parser.add_argument(
        "--ddp-port",
        type=str,
        default="11111",
        help="Port to use for distributed data parallel.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force using cpu (for debugging, overwrite force_cpu in config).",
    )

    parser.add_argument(
        "--reverse-gt",
        action="store_true",
        help="Whether to reverse text sequence in ground truth (for arabic language for instance).",
    )

    parser.add_argument(
        "--dataset",
        required=False,
        help="Overwrite train/valid/test dataset in config.",
        type=str
    )

    parser.add_argument(
        "--dataset-config",
        required=False,
        help="Load complex dataset configuration (useful for training with multiple datasets).",
        type=str
    )

    
    parser.add_argument(
        "--dataset-same-sampler",
        help="Use same number of training samples for each training dataset (useful for training with multiple datasets).",
        action="store_true",
    )

    parser.add_argument(
        "--load-images-on-the-fly",
        action="store_true",
        help="Whether to load images on the fly or load them in memory directly.",
    )

    parser.add_argument(
        "--transfer",
        required=False,
        help="Path of checkpoint to use for transfer learning (overwrite transfer learning in config).",
    )

    parser.add_argument(
        "--transfer-keyword",
        default="best",
        help="If transfer learning is specified by folder, specify a keyword that is in the checkpoint file name to load.",
    )

    parser.add_argument(
        "--transfer-encoder",
        required=False,
        help="Path/url of checkpoint to use for transfer learning for encoder only (overwrite transfer learning in config).",
    )

    parser.add_argument(
        "--transfer-decoder",
        required=False,
        help="Path/url of checkpoint to use for transfer learning for decoder only (overwrite transfer learning in config).",
    )

    parser.add_argument(
        "--encoder-dropout",
        required=False,
        type=float,
        help="Dropout ratio to use for encoder (overwrite enc_dropout in config).",
    )

    parser.add_argument(
        "--decoder-dropout",
        required=False,
        type=float,
        help="Dropout ratio to use for decoder (overwrite dec_dropout in config).",
    )

    parser.add_argument(
        "--checkpoint",
        required=False,
        help="Path of checkpoint (file or folder) to use for predict mode.",
    )

    parser.add_argument(
        "--reset-optimizer-after-k-samples",
        required=False,
        type=int,
        help="Reset optimizer with initial learning rate, removing computation history after processing k training samples.",
    )

    parser.add_argument(
        "--log-gradients",
        required=False,
        action="store_true",
        help="Add log about gradient for each backpropagation step in tensorboard file.",
    )

    parser.add_argument(
        "--training-time",
        required=False,
        help="Overwrite max_training_time in config.",
    )

    parser.add_argument(
        "--gradscaler-init-value",
        required=False,
        default=65536.0,
        type=int,
        help="Init value for gradscaler (when using amp).",
    )

    parser.add_argument(
        "--lr-all",
        help="Overwrite the learning rate value in config for all models (encoder+decoder).",
        type=float
    )

    parser.add_argument(
        "--lr-encoder",
        help="Overwrite the learning rate value in config for the encoder only.",
        type=float
    )

    parser.add_argument(
        "--lr-decoder",
        help="Overwrite the learning rate value in config for the decoder only.",
        type=float
    )

    parser.add_argument(
        "--decoding-mt-mode",
        choices=["static", "dynamic"],
        default="static",
        help="For mt_dan at inference time only. Static = always use the k predicted tokens. Dynamic = use the prediction at t+i only if above a given threshold.",
    )

    parser.add_argument(
        "--decoding-win-mode",
        choices=["first", "average", "majority"],
        default="first",
        help="For meta_dan at inference time only, when several predictions concern the same token."
             "first = always use the prediction of the first head."
             "average = average the scores regarding the same token prediction before softmax."
             "majority = use majority vote to select the final prediction.",
    )

    parser.add_argument(
        "--decoding-threshold",
        type=float,
        help="Percent threshold to validate a prediction at t+i (to combine with --decoding-mode='dynamic')",
    )

    parser.add_argument(
        "--decoding-num-preds",
        type=int,
        help="Fixed number of predictions to consider as valids at each decoding iteration (to combine with --decoding-mt-mode='static')",
    )

    parser.add_argument(
        "--pSource",
        required=False,
        help="File/Folder path containing images for predict mode.",
    )

    parser.add_argument(
        "--pTarget",
        required=False,
        help="Folder path to store prediction for predict mode.",
    )

    parser.add_argument(
        "--pAttention",
        action="store_true",
        help="Output attention video in addition to prediction (for predict mode only).",
    )

    parser.add_argument(
        "--output-folder",
        required=False,
        help="Overwrite output_folder in config.",
    )

    parser.add_argument(
        "--token-win",
        required=False,
        type=int,
        help="Overwrite token_win in config (for w_dan and meta_dan only).",
    )

    parser.add_argument(
        "--use-one-head-per-token-pos",
        required=False,
        action="store_true",
        help="Use one projection head per position in token window (for w-dan only)",
    )

    parser.add_argument(
        "--num-token-heads",
        required=False,
        type=int,
        help="Overwrite num_token_heads in config (for mt_dan and meta_dan only).",
    )

    parser.add_argument(
        "--use-query-embedding",
        action="store_true",
        help="Overwrite use_query_embedding in config (for w_dan).",
    )

    parser.add_argument(
        "--max-char-prediction",
        required=False,
        type=int,
        help="Overwrite max_char_prediction in config (for dan-like model).",
    )

    parser.add_argument(
        "--tokenizer",
        default="character",
        help="Tokenizer to use.",
        choices=['character', 'bpe'],
    )

    parser.add_argument(
        "--tok-model-name",
        help="Tokenizer model (file name) to use.",
    )

    parser.add_argument(
        "--tokenizer-source",
        default="dataset",
        help="Source to train tokenizer (for bpe)",
        choices=['dataset',
                 'wiki-en',  # English
                 'wiki-fr',  # French
                 'wiki-de',  # German
                 'wiki-nl',  # Nederland
                 'wiki-ru',  # Russian
                 'wiki-es',  # Spanish
                 'wiki-zh',  # Chinese
                 'wiki-ar',  # Arabic
                 ],
    )

    parser.add_argument(
        "--tok-num-tokens",
        default=1000,
        help="Vocabulary size for tokenizer (for bpe)",
        type=int
    )

    parser.add_argument(
        "--encoder",
        default="fcn-m",
        help="Encoder architecture to use.",
        choices=['fcn', 'fcn-m', 'convnext-v2-b', 'convnext-v2-t', 'convnext-v2-n'],
    )

    parser.add_argument(
        "--adapt-encoder-downscaling",
        action="store_true",
        help="Whether to adapt the pooling/stride to have a height downscaling factor of 32 and a width downscaling factor of 8",
    )

    parser.add_argument(
        "--convnext-no-end-norm",
        action="store_true",
        help="Prevent the final normalization to be applied in convnext encoder",
    )

    parser.add_argument(
        "--transfer-freeze-encoder",
        action="store_true",
        help="Prevent weight update for the encoder (except last layer).",
    )

    parser.add_argument(
        "--syn-config",
        required=False,
        help="Overwrite synthetic_data in config (config file name for synthetic data).",
    )

    parser.add_argument(
        "--no-syn",
        action="store_true",
        help="Overwrite synthetic_data with None in config.",
    )

    parser.add_argument(
        "--encoder-post-norm",
        action="store_true",
        help="Add instance norm at the end of encoder block.",
    )

    parser.add_argument(
        "--syn-mode",
        required=False,
        choices=['line', 'text-block', 'page', ],
        help="Overwrite mode in syn config.",
    )

    parser.add_argument(
        "--syn-double-line",
        required=False,
        action="store_true",
        help="Concatenate ground truth of two line samples for synthetic line generation.",
    )

    parser.add_argument(
        "--syn-single-column",
        required=False,
        action="store_true",
        help="Generate only single column synthetic pages..",
    )

    parser.add_argument(
        "--syn-custom-config",
        required=False,
        help="Useful for dataset-specific synthetic document generation ('generic_page' is default).",
    )


    parser.add_argument(
        "--syn-source",
        default="dataset",
        choices=['random',
                 'dataset',
                 'wiki-en',  # English
                 'wiki-fr',  # French
                 'wiki-de',  # German
                 'wiki-nl',  # Nederland
                 'wiki-ru',  # Russian
                 'wiki-es',  # Spanish
                 'wiki-zh',  # Chinese
                 'wiki-ar',  # Arabic
                 ],
        help="Source of text to generate synthetic samples.",
    )

    parser.add_argument(
        "--syn-num-fonts",
        default=1000,
        help="Number of fonts to use for synthetic document generation.",
        type=int
    )

    parser.add_argument(
        "--syn-init-proba",
        help="Probability to generate synthetic data when training starts (overwrite init_proba in syn_config).",
        type=float
    )

    parser.add_argument(
        "--syn-end-proba",
        help="Probability to generate synthetic data after main training stage (overwrite end_proba in syn_config).",
        type=float
    )

    parser.add_argument(
        "--syn-num-samples",
        help="Number of samples for scheduler for synthetic/real data ratio to go from init_proba to end_proba "
             "(overwrite num_samples_proba in syn_config).",
        type=int
    )

    parser.add_argument(
        "--curr-num-samples",
        help="Number of samples for curriculum learning update (overwrite num_samples_curr in syn_config).",
        type=int
    )

    parser.add_argument(
        "--start-curr-at-sample",
        help="Number of samples before curriculum learning starts (overwrite start_curr_at_sample in syn_config).",
        type=int
    )

    parser.add_argument(
        "--num-workers",
        help="Number of workers for dataloader.",
        type=int
    )

    parser.add_argument(
        "--image-padding-value",
        help="Padding value to batch images (float value between 0 and 1, or 'mean').",
    )

def get_parser():
    parser = argparse.ArgumentParser(prog="dar")

    subparsers = parser.add_subparsers(dest="mode")

    add_format_dataset_parser(subparsers)
    add_model_parser(subparsers)
    return parser


def check_args(parser, args):
    if args["mode"] == "predict" and args["checkpoint"] is None:
        parser.error("--checkpoint must be specified for predict mode.")
    if args["mode"] == "predict" and args["pSource"] is None:
        parser.error("--pSource must be specified for predict mode.")
    if args["mode"] == "predict" and args["pSource"] is not None and args["pTarget"] is None:
        args["pTarget"] = os.path.dirname(args["pSource"])
    if args["mode"] in ["train", "eval"] and args["config"] is None:
        args["config"] = DEFAULT_CONFIG[args["network"]]
    if args["transfer"] and (args["transfer_encoder"] or args["transfer_decoder"]):
        parser.error("--transfer is mutually exclusive with --transfer-encoder and --transfer-decoder")


def overwrite_params(params, args):
    for batch_key in ["train_batch_size", "val_batch_size", "test_batch_size"]:
        if args[batch_key]:
            params["training_params"][batch_key] = args[batch_key]
    if args["all_batch_size"]:
        for batch_key in ["train_batch_size", "val_batch_size", "test_batch_size"]:
            params["training_params"][batch_key] = args["all_batch_size"]
    if args["reset_optimizer_after_k_samples"]:
        params["training_params"]["reset_optimizer_after_k_samples"] = args["reset_optimizer_after_k_samples"]
    assert not ((args["lr_all"] and (args["lr_encoder"] or args["lr_decoder"])) or \
                ((args["lr_encoder"] is not None) ^ (args["lr_decoder"] is not None))), \
        "--lr-encoder and --lr-decoder must be jointly specified and are mutually exclusive with --lr-all"
    if args["lr_all"]:
        params["training_params"]["optimizers"]["all"]["args"]["lr"] = args["lr_all"]
    if (args["lr_encoder"] is not None) and (args["lr_decoder"] is not None):
        default_config = params["training_params"]["optimizers"]["all"]
        params["training_params"]["optimizers"] = {
            "encoder": copy.deepcopy(default_config),
            "decoder": copy.deepcopy(default_config)
        }
        params["training_params"]["optimizers"]["encoder"]["args"]["lr"] = args["lr_encoder"]
        params["training_params"]["optimizers"]["decoder"]["args"]["lr"] = args["lr_decoder"]
    if args["no_amp"]:
        params["training_params"]["use_amp"] = False
    if args["ddp"]:
        params["training_params"]["use_ddp"] = True
    if args["ddp_port"]:
        params["training_params"]["ddp_port"] = args["ddp_port"]
    if args["cpu"]:
        params["training_params"]["force_cpu"] = True
    if args["max_num_epochs"]:
        params["training_params"]["max_nb_epochs"] = args["max_num_epochs"]
    if args["max_num_samples"]:
        params["training_params"]["max_num_samples"] = args["max_num_samples"]
    if args["max_char_prediction"]:
        params["training_params"]["max_char_prediction"] = args["max_char_prediction"]
    if args["grad_acc"]:
        params["training_params"]["gradient_accumulation"] = args["grad_acc"]
    if args["grad_clip_max"] is not None:
        params["training_params"]["gradient_clipping"] = {
            "models": ["encoder", "decoder"],
            "max": args["grad_clip_max"],
        }
    params["training_params"]["log_gradients"] = args["log_gradients"]
    if args["add_weights_overlap_penalty"]:
        params["training_params"]["add_weights_overlap_penalty"] = True
        params["training_params"]["train_metrics"].append("loss_weights")
    if args["add_loop_teacher_forcing"]:
        params["training_params"]["add_loop_teacher_forcing"] = True
    if args["weights_penalty_factor"] is not None:
        params["training_params"]["weights_penalty_factor"] = args["weights_penalty_factor"]
    if args["output_folder"]:
        params["training_params"]["output_folder"] = args["output_folder"]
    if args["training_time"]:
        params["training_params"]["max_training_time"] = float(args["training_time"])
    if args["loss_reduction"]:
        params["training_params"]["loss_reduction"] = args["loss_reduction"]
    if args["save_mode"]:
        params["training_params"]["save_weights"]["mode"] = args["save_mode"]
    if args["save_interval"]:
        params["training_params"]["save_weights"]["interval"] = args["save_interval"]
    if args["eval_mode"]:
        params["training_params"]["eval_on_valid"]["mode"] = args["eval_mode"]
    if args["eval_interval"]:
        params["training_params"]["eval_on_valid"]["interval"] = args["eval_interval"]
    if args["eval_start"]:
        params["training_params"]["eval_on_valid"]["start_at"] = args["eval_start"]
    if args["transfer_keyword"]:
        params["model_params"]["transfer_key"] = args["transfer_keyword"]
    if args["transfer"]:
        params["model_params"]["transfer_learning"] = {
                "encoder": ["encoder", args["transfer"], True, True],
                "decoder": ["decoder", args["transfer"], True, False],
            }
    if args["transfer_encoder"]:
        params["model_params"]["transfer_learning"] = {
                "encoder": ["encoder", args["transfer_encoder"], True, False],
        }
    if args["transfer_decoder"]:
        if params["model_params"]["transfer_learning"]:
            params["model_params"]["transfer_learning"]["decoder"] = ["decoder", args["transfer_decoder"], True, False]
        else:
            params["model_params"]["transfer_learning"] = {
                    "decoder": ["decoder", args["transfer_decoder"], True, False],
            }
    if args["transfer_freeze_encoder"]:
        params["model_params"]["transfer_learning"]["encoder"][2] = False
    if args["encoder_dropout"] is not None:
        params["model_params"]["enc_dropout"] = args["encoder_dropout"]
    if args["decoder_dropout"] is not None:
        params["model_params"]["dec_dropout"] = args["decoder_dropout"]
    if args["encoder_post_norm"]:
        params["model_params"]["encoder_post_norm"] = args["encoder_post_norm"]
    if args["convnext_no_end_norm"]:
        params["model_params"]["convnext_no_end_norm"] = args["convnext_no_end_norm"]
    if args["drop_last_res"]:
        params["model_params"]["dec_drop_last_res"] = args["drop_last_res"]
    if args["drop_all_res"]:
        params["model_params"]["dec_drop_all_res"] = args["drop_all_res"]
    if args["decoding_win_mode"]:
        params["model_params"]["decoding_win_mode"] = args["decoding_win_mode"]
    if args["decoding_mt_mode"]:
        params["model_params"]["decoding_mt_mode"] = args["decoding_mt_mode"]
    params["model_params"]["use_one_head_per_token_pos"] = args["use_one_head_per_token_pos"]
    if args["decoding_threshold"]:
        params["model_params"]["decoding_threshold"] = args["decoding_threshold"]
    if args["decoding_num_preds"]:
        params["model_params"]["decoding_num_preds"] = args["decoding_num_preds"]
    if args["token_win"]:
        params["model_params"]["token_win"] = args["token_win"]
    if args["num_token_heads"]:
        params["model_params"]["num_token_heads"] = args["num_token_heads"]
    if args["use_query_embedding"]:
        params["model_params"]["use_query_embedding"] = args["use_query_embedding"]
    params["model_params"]["adapt_encoder_downscaling"] = args["adapt_encoder_downscaling"]
    if args["image_padding_value"]:
        if args["image_padding_value"] in ["mean", ]:
            params["dataset_params"]["config"]["image_padding_value"] = args["image_padding_value"]
        else:
            params["dataset_params"]["config"]["image_padding_value"] = float(args["image_padding_value"])
    if args["reverse_gt"]:
        params["dataset_params"]["config"]["reverse_gt"] = True
    params["dataset_params"]["config"]["dataset_same_sampler"] = args["dataset_same_sampler"]
    if args["syn_config"]:
        params["dataset_params"]["config"]["synthetic_data"] = args["syn_config"]
    if args["syn_source"]:
        params["dataset_params"]["config"]["syn_source"] = args["syn_source"]
    if args["add_loop_syn"]:
        params["dataset_params"]["config"]["add_loop_syn"] = True
    if args["syn_num_fonts"]:
        params["dataset_params"]["config"]["syn_num_fonts"] = args["syn_num_fonts"]
    if args["tokenizer_source"]:
        params["dataset_params"]["tokenizer_source"] = args["tokenizer_source"]
    if args["tok_num_tokens"] is not None:
        params["dataset_params"]["tokenizer_num_tokens"] = args["tok_num_tokens"]
    params["dataset_params"]["tokenizer_model_name"] = args["tok_model_name"]
    if args["load_images_on_the_fly"] is not None:
        params["dataset_params"]["config"]["load_in_memory"] = not args["load_images_on_the_fly"]
    if args["syn_init_proba"] is not None:
        params["dataset_params"]["config"]["syn_init_proba"] = args["syn_init_proba"]
    if args["syn_end_proba"] is not None:
        params["dataset_params"]["config"]["syn_end_proba"] = args["syn_end_proba"]
    if args["syn_single_column"] is not None:
        params["dataset_params"]["config"]["syn_single_column"] = args["syn_single_column"]
    if args["syn_num_samples"] is not None:
        params["dataset_params"]["config"]["syn_num_samples"] = args["syn_num_samples"]
    if args["curr_num_samples"] is not None:
        params["dataset_params"]["config"]["curr_num_samples"] = args["curr_num_samples"]
    if args["start_curr_at_sample"] is not None:
        params["dataset_params"]["config"]["start_curr_at_sample"] = args["start_curr_at_sample"]
    params["dataset_params"]["config"]["track_hist_tokens"] = args["track_hist_tokens"]
    if args["syn_custom_config"]:
        params["dataset_params"]["config"]["syn_custom_config"] = args["syn_custom_config"]
    if args["syn_mode"]:
        params["dataset_params"]["config"]["syn_mode"] = args["syn_mode"]
    params["dataset_params"]["config"]["syn_double_line"] = args["syn_double_line"]
    if args["num_workers"]:
        params["dataset_params"]["config"]["worker_per_gpu"] = args["num_workers"]
    if params["args"]["tokenizer"] == "bpe":
        os.environ['TOKENIZERS_PARALLELISM'] = "false"

    params["training_params"]["gradscaler_init_value"] = args["gradscaler_init_value"]
    params["training_params"]["seed"] = args["seed"]
    if args["no_syn"]:
        params["dataset_params"]["config"]["synthetic_data"] = None
    assert not (args["syn_custom_config"] and args["dataset_config"])
    return params


def main():
    parser = get_parser()
    args = vars(parser.parse_args())

    if args["mode"] == "format":
        run_format_dataset(parser)
    else:
        run_model(parser)


def run_format_dataset(parser):
    args = vars(parser.parse_args())
    if "func" in args:
        args.pop("func")(**args)
    else:
        parser.print_help()


def run_model(parser):
    args = vars(parser.parse_args())
    check_args(parser, args)
    params = {
        "args": args,
    }
    if args["config"]:
        params.update(load_yaml_file(os.path.join("configs/", args["config"])))
        params = overwrite_params(params, args)
        params = preprocess_params(params)
    run(params)


if __name__ == '__main__':
    main()
