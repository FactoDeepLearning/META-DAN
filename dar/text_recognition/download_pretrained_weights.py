"""
Module defining functions related to online weight downloading.
"""

import wget
import os
import sys
import torch

# Association str from argparse to weight url
pretrained_model_weights = {
    "convnext-v2-t-in22k": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt",
    "convnext-v2-b-in22k": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt",
    "convnext-v2-n-in22k": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_384_ema.pt",
    "convnext-v2-t-mae": "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_tiny_1k_224_fcmae.pt",
    "convnext-v2-b-mae": "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_base_1k_224_fcmae.pt",
    "convnext-v2-n-mae": "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_nano_1k_224_fcmae.pt",

    # pre-trained models at line level
    "line_ctc_convnext-v2-b_latin": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/convnext-v2-b_latin_line.pt",
    "line_ctc_convnext-v2-n_latin": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/convnext-v2-n_latin_line.pt",
    "line_ctc_convnext-v2-t_latin": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/convnext-v2-t_latin_line.pt",
    "line_ctc_fcn_m_bressay": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/fcn_m_bressay_line.pt",
    "line_ctc_fcn_m_casia2": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/fcn_m_casia2_line.pt",
    "line_ctc_fcn_m_eparchos": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/fcn_m_eparchos_line.pt",
    "line_ctc_fcn_m_esposalles": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/fcn_m_esposalles_line.pt",
    "line_ctc_fcn_m_iam": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/fcn_m_iam_line.pt",
    "line_ctc_fcn_m_latin": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/fcn_m_latin_line.pt",
    "line_ctc_fcn_m_maurdor_c3": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/fcn_m_maurdor_c3_line_full_syn.pt",
    "line_ctc_fcn_m_maurdor_c4": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/fcn_m_maurdor_c4_line_full_syn.pt",
    "line_ctc_fcn_m_read2016": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/fcn_m_read2016_line.pt",
    "line_ctc_fcn_m_rimes2009": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/fcn_m_rimes2009_line_full_syn.pt",
    "line_ctc_fcn_m_rimes2024": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/fcn_m_rimes2024_line_full_syn.pt",
    "line_ctc_fcn_m_scribblelens": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/fcn_line/fcn_m_scribblelens_line.pt",

    # pre-trained models for meta-dan
    "meta_dan_convnext-v2-b_latin": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_convnext-v2-b_latin_page.pt",
    "meta_dan_convnext-v2-n_latin": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_convnext-v2-n_latin_page.pt",
    "meta_dan_convnext-v2-t_latin": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_convnext-v2-t_latin_page.pt",
    "meta_dan_fcn_m_bressay": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_fcn_m_bressay_page.pt",
    "meta_dan_fcn_m_casia2": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_fcn_m_casia2_page.pt",
    "meta_dan_fcn_m_eparchos": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_fcn_m_eparchos_page.pt",
    "meta_dan_fcn_m_esposalles": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_fcn_m_esposalles_page.pt",
    "meta_dan_fcn_m_iam": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_fcn_m_iam_page.pt",
    "meta_dan_fcn_m_latin": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_fcn_m_latin_page.pt",
    "meta_dan_fcn_m_maurdor_c3": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_fcn_m_maurdor_c3_page.pt",
    "meta_dan_fcn_m_maurdor_c4": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_fcn_m_maurdor_c4_page.pt",
    "meta_dan_fcn_m_read2016": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_fcn_m_read2016_page.pt",
    "meta_dan_fcn_m_rimes2009": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_fcn_m_rimes2009_page.pt",
    "meta_dan_fcn_m_rimes2024": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_fcn_m_rimes2024_page.pt",
    "meta_dan_fcn_m_scribblelens": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/meta_dan/meta_dan_fcn_m_scribblelens_page.pt",  


    # pre-trained models for dan
    "dan_fcn_m_bressay": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/dan/dan_fcn_m_bressay_page.pt",
    "dan_fcn_m_casia2": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/dan/dan_fcn_m_casia2_page.pt",
    "dan_fcn_m_eparchos": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/dan/dan_fcn_m_eparchos_page.pt",
    "dan_fcn_m_esposalles": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/dan/dan_fcn_m_esposalles_page.pt",
    "dan_fcn_m_iam": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/dan/dan_fcn_m_iam_page.pt",
    "dan_fcn_m_maurdor_c3": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/dan/dan_fcn_m_maurdor_c3_page.pt",
    "dan_fcn_m_maurdor_c4": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/dan/dan_fcn_m_maurdor_c4_page.pt",
    "dan_fcn_m_read2016": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/dan/dan_fcn_m_read2016_page.pt",
    "dan_fcn_m_rimes2009": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/dan/dan_fcn_m_rimes2009_page.pt",
    "dan_fcn_m_rimes2024": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/dan/dan_fcn_m_rimes2024_page.pt",
    "dan_fcn_m_scribblelens": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/dan/dan_fcn_m_scribblelens_page.pt", 

    # pre-trained models for faster-dan
    "faster_dan_fcn_m_bressay": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/faster_dan/faster_dan_faster_dan_fcn_m_bressay_page.pt",
    "faster_dan_fcn_m_casia2": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/faster_dan/faster_dan_fcn_m_casia2_page.pt",
    "faster_dan_fcn_m_eparchos": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/faster_dan/faster_dan_fcn_m_eparchos_page.pt",
    "faster_dan_fcn_m_esposalles": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/faster_dan/faster_dan_fcn_m_esposalles_page.pt",
    "faster_dan_fcn_m_iam": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/faster_dan/faster_dan_fcn_m_iam_page.pt",
    "faster_dan_fcn_m_maurdor_c3": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/faster_dan/faster_dan_fcn_m_maurdor_c3_page.pt",
    "faster_dan_fcn_m_maurdor_c4": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/faster_dan/faster_dan_fcn_m_maurdor_c4_page.pt",
    "faster_dan_fcn_m_read2016": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/faster_dan/faster_dan_fcn_m_read2016_page.pt",
    "faster_dan_fcn_m_rimes2009": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/faster_dan/faster_dan_fcn_m_rimes2009_page.pt",
    "faster_dan_fcn_m_rimes2024": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/faster_dan/faster_dan_fcn_m_rimes2024_page.pt",
    "faster_dan_fcn_m_scribblelens": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/faster_dan/faster_dan_fcn_m_scribblelens_page.pt",  

    # pre-trained models for mt-dan
    "mt_dan_fcn_m_bressay": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/mt_dan/mt_dan_mt_dan_fcn_m_bressay_page.pt",
    "mt_dan_fcn_m_casia2": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/mt_dan/mt_dan_fcn_m_casia2_page.pt",
    "mt_dan_fcn_m_eparchos": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/mt_dan/mt_dan_fcn_m_eparchos_page.pt",
    "mt_dan_fcn_m_esposalles": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/mt_dan/mt_dan_fcn_m_esposalles_page.pt",
    "mt_dan_fcn_m_iam": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/mt_dan/mt_dan_fcn_m_iam_page.pt",
    "mt_dan_fcn_m_maurdor_c3": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/mt_dan/mt_dan_fcn_m_maurdor_c3_page.pt",
    "mt_dan_fcn_m_maurdor_c4": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/mt_dan/mt_dan_fcn_m_maurdor_c4_page.pt",
    "mt_dan_fcn_m_read2016": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/mt_dan/mt_dan_fcn_m_read2016_page.pt",
    "mt_dan_fcn_m_rimes2009": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/mt_dan/mt_dan_fcn_m_rimes2009_page.pt",
    "mt_dan_fcn_m_rimes2024": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/mt_dan/mt_dan_fcn_m_rimes2024_page.pt",
    "mt_dan_fcn_m_scribblelens": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/mt_dan/mt_dan_fcn_m_scribblelens_page.pt",  

    # pre-trained models for w-dan
    "w_dan_fcn_m_bressay": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/w_dan/w_dan_w_dan_fcn_m_bressay_page.pt",
    "w_dan_fcn_m_casia2": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/w_dan/w_dan_fcn_m_casia2_page.pt",
    "w_dan_fcn_m_eparchos": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/w_dan/w_dan_fcn_m_eparchos_page.pt",
    "w_dan_fcn_m_esposalles": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/w_dan/w_dan_fcn_m_esposalles_page.pt",
    "w_dan_fcn_m_iam": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/w_dan/w_dan_fcn_m_iam_page.pt",
    "w_dan_fcn_m_maurdor_c3": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/w_dan/w_dan_fcn_m_maurdor_c3_page.pt",
    "w_dan_fcn_m_maurdor_c4": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/w_dan/w_dan_fcn_m_maurdor_c4_page.pt",
    "w_dan_fcn_m_read2016": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/w_dan/w_dan_fcn_m_read2016_page.pt",
    "w_dan_fcn_m_rimes2009": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/w_dan/w_dan_fcn_m_rimes2009_page.pt",
    "w_dan_fcn_m_rimes2024": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/w_dan/w_dan_fcn_m_rimes2024_page.pt",
    "w_dan_fcn_m_scribblelens": "https://huggingface.co/FactoDeepLearning/MetaDAN/resolve/main/w_dan/w_dan_fcn_m_scribblelens_page.pt",  
}


def progress_bar(current: int, total: int, width:int=80) -> None:
  """
  Showing progress bar for weight downloading.
  :param current: number of downloaded bytes.
  :param total: totoal number of bytes to download (weight file size)
  """
  progress_message = f"Downloading: {current / total * 100}% [{current} / {total}] bytes"
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()


def download(url: str, dir: str=".cache", file_name: str=None) -> None:
    """
    Download weight located at url and save it as file_name at dir path.
    """
    file_name = os.path.basename(url) if file_name is None else file_name
    out_path = os.path.join(dir, file_name)
    print(f"Download {url}; target: {out_path}")
    wget.download(url, out_path, bar=progress_bar)


def check_exist_or_download_model(model_name: str, dir: str=".cache") -> str:
    """
    Check if target model weights have already been downloaded, do it if not already done.
    :param model_name: key for pretrained_model_weights variable
    :param dir: directory path in which to download/look for weights.
    """
    file_name = model_name + ".pt"
    file_path = os.path.join(dir, file_name)
    if not os.path.exists(file_path):
        if model_name in pretrained_model_weights:
            download(pretrained_model_weights[model_name], dir, file_name)
            convert_checkpoint(os.path.join(dir, file_name), model_name)
    return file_path


def check_exist_or_download_checkpoint(model_name_or_checkpoint_path: str) -> str:
    """
    If argument corresponds to a model name, download its weights and return correpsonding path. Otherwise, it correpsonds to a local file path (return itself).
    """
    if not os.path.exists(model_name_or_checkpoint_path):
        return check_exist_or_download_model(model_name_or_checkpoint_path)
    return model_name_or_checkpoint_path


def convert_checkpoint(checkpoint_path: str, model_name: str) -> None:
    """
    Convert out-of-package model checkpoints to make it compatible with local scripts.
    :param checkpoint_path: path of checkpoint file.
    :param model_name: corresponding key for pretrained_model_weights variable.
    """
    if "convnext" in model_name:
        checkpoint = torch.load(checkpoint_path)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            checkpoint["encoder_state_dict"] = state_dict
            del checkpoint["model"]
            torch.save(checkpoint, checkpoint_path)
