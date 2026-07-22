import os.path

from dar.text_recognition.line.ctc.trainer import LineCTCManager
from dar.text_recognition.line.ctc.models import CTCDecoder
from dar.text_recognition.document.w_dan.trainer import WDANManager
from dar.text_recognition.document.w_dan.models import WDANDecoder
from dar.text_recognition.document.meta_dan.models import MetaDANDecoder
from dar.text_recognition.document.meta_dan.trainer import MetaDANManager
from dar.text_recognition.document.mt_dan.trainer import MTDANManager
from dar.text_recognition.document.mt_dan.models import MTDANDecoder
from dar.text_recognition.document.faster_dan.trainer import FasterDANManager
from dar.text_recognition.document.faster_dan.models import FasterDANDecoder
from dar.text_recognition.document.dan.trainer import DANManager
from dar.text_recognition.document.dan.models import DANDecoder
from dar.text_recognition.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from dar.text_recognition.preprocessings import text_preprocessing
from dar.text_recognition.tokenizer import CharacterTokenizer, BPETokenizer
from dar.generic.models import FCN_Encoder, FCN_Encoder_M256
from dar.generic.utils import set_deterministic, find_file_by_keyword
from dar.external.ConvNeXtV2.convnext_model import ConvNeXtV2Encoder
from dar.text_recognition.download_pretrained_weights import check_exist_or_download_checkpoint
import torch
import torch.multiprocessing as mp
import numpy as np

# Associate str argument to encoder class
ENCODERS = {
    "fcn": FCN_Encoder,
    "fcn-m": FCN_Encoder_M256,
    "convnext-v2-b": ConvNeXtV2Encoder,
    "convnext-v2-t": ConvNeXtV2Encoder,
    "convnext-v2-n": ConvNeXtV2Encoder
}

# Associate str argument to trainer and decoder classes
CONFIG = {
    "line_ctc": {
        "manager": LineCTCManager,
        "decoder": CTCDecoder
    },
    "dan": {
        "manager": DANManager,
        "decoder": DANDecoder
    },
    "faster_dan": {
        "manager": FasterDANManager,
        "decoder": FasterDANDecoder
    },    
    "w_dan": {
        "manager": WDANManager,
        "decoder": WDANDecoder
    },
    "mt_dan": {
        "manager": MTDANManager,
        "decoder": MTDANDecoder
    },
    "meta_dan": {
        "manager": MetaDANManager,
        "decoder": MetaDANDecoder
    },

}


def start(rank, params):
    """
    Main function for training/evaluation
    """
    set_deterministic(params["args"]["seed"])
    net_name = params["args"]["network"]
    params["training_params"]["ddp_rank"] = params["training_params"]["gpus"][rank]
    params["dataset_params"]["dataset_manager"] = OCRDatasetManager
    params["dataset_params"]["dataset_class"] = OCRDataset
    params["dataset_params"]["tokenizer_class"] = CharacterTokenizer if params["args"]["tokenizer"] == "character" else BPETokenizer
    params["dataset_params"]["config"]["preprocessing"] = text_preprocessing
    params["model_params"]["models"] = {
        "encoder": ENCODERS[params["args"]["encoder"]],
        "decoder": CONFIG[net_name]["decoder"]
    }
    params["dataset_params"]["mode"] = params["args"]["mode"]

    # Handle convnext encoder
    params["dataset_params"]["config"]["round_function"] = np.floor if "convnext" in params["args"]["encoder"] else np.ceil

    if "convnext" in params["args"]["encoder"] and not params["args"]["adapt_encoder_downscaling"]:
        params["dataset_params"]["config"].update({
            "width_divisor": 32,
            "height_divisor": 32
        })
    if params["args"]["encoder"] == "convnext-v2-t":
        params["model_params"].update({
            "depths": [3, 3, 9, 3],
            "dims": [96, 192, 384, 768]
        })
    elif params["args"]["encoder"] == "convnext-v2-b":
        params["model_params"].update({
            "depths": [3, 3, 27, 3],
            "dims": [128, 256, 512, 1024]
        })
    elif params["args"]["encoder"] == "convnext-v2-n":
        params["model_params"].update({
            "depths": [2, 2, 8, 2],
            "dims": [80, 160, 320, 640]
        })


    model = CONFIG[net_name]["manager"](params)

    # Training process
    if params["args"]["mode"] == "train":
        model.params["training_params"]["load_epoch"] = "last"
        model.load_model()
        model.train()

    # evaluation process
    elif params["args"]["mode"] == "eval":
        model.params["training_params"]["load_epoch"] = params["args"]["eval_weight_key"]
        model.load_model()

        model.params["training_params"]["eval_metrics"].append("pred")
        for dataset in params["dataset_params"]["test"]:
            model.evaluate(dataset["name"], split_name="test", output=True)
        for dataset in params["dataset_params"]["val"]:
            model.evaluate(dataset["name"], split_name="val", output=True)

    # evaluation processfor mt_dan only (try all decoding parameter static/dynamic)
    elif params["args"]["mode"] == "eval_all":
        model.params["training_params"]["load_epoch"] = params["args"]["eval_weight_key"]
        model.load_model()
        model.eval_all()

    if params["training_params"]["use_ddp"]:
        model.cleanup()


def predict(params):
    """
    Main function for prediction (without ground truth)
    """
    net_name = params["args"]["network"]
    
    if os.path.isdir(params["args"]["checkpoint"]):
        checkpoint_path = find_file_by_keyword(params["args"]["checkpoint"], "best")
    else:
        checkpoint_path = params["args"]["checkpoint"]
        checkpoint_path = check_exist_or_download_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_params = checkpoint["params"]
    model = CONFIG[net_name]["manager"](checkpoint_params, load_dataset=False)
    model.load_model(checkpoint=checkpoint)
    for key in ["decoding_mt_mode", "decoding_threshold", "decoding_num_preds", "decoding_win_mode", "first_pass_decoding_threshold", "first_pass_decoding_mt_mode", "first_pass_decoding_num_preds"]:
        if "model_params" in params and key in params["model_params"]:
            model.params["model_params"][key] = params["model_params"][key]
    model.predict(params["args"]["pSource"], params["args"]["pTarget"], params["args"]["pAttention"])


def run(params):
    """
    Main function that manage training, evaluation and prediction
    """
    if params["args"]["mode"] == "predict":
        predict(params)
    elif params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
        mp.spawn(start, args=(params, ), nprocs=len(params["training_params"]["gpus"]), join=True)
    else:
        start(0, params)
