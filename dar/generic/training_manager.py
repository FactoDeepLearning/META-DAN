"""
Module defining generic class to handle training, evaluation and prediction 
"""

import torch
import os
import sys
import copy
import json
import shutil
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.init import kaiming_uniform_
from torch.nn import Module
from torch import Tensor
from tqdm import tqdm
from time import time
from dar.generic.ddp_utils import DataParallelPassthrough as DDP
from dar.generic.ddp_utils import convert_checkpoint_from_ddp
from torch import GradScaler
from dar.generic.metric_manager import MetricManager
from dar.generic.scheduler import DropoutScheduler
from dar.generic.utils import find_file_by_keyword
from dar.text_recognition.download_pretrained_weights import check_exist_or_download_checkpoint
from datetime import date
import itertools


class GenericTrainingManager:

    def __init__(self, params: dict, load_dataset: bool=True):
        self.type = None
        self.is_master = False
        self.params = params
        self.dropout_scheduler = None
        self.models = {}
        self.begin_time = None
        self.dataset = None
        self.tokenizer = None
        self.paths = None

        #for gradient accumulation
        self.grad_acc_step = 0
        self.grad_acc_ind = 0

        self.latest_step = 0
        self.latest_epoch = -1
        self.latest_batch = 0
        self.latest_sample = 0
        self.total_batch = 0
        self.grad_acc_step = 0
        self.latest_train_metrics = dict()
        self.latest_valid_metrics = dict()
        self.curriculum_info = dict()
        self.curriculum_info["latest_valid_metrics"] = dict()
        self.max_training_time = 3600*self.params["training_params"]["max_training_time"]
        self.phase = None
        self.max_mem_usage_by_epoch = list()
        self.losses = list()
        self.lr_values = list()

        self.scaler = None
        self.device = None
        self.autocast_mode = "cpu"

        self.optimizers = dict()
        self.optimizers_named_params_by_group = dict()
        self.lr_schedulers = dict()
        self.best = None
        self.writer = None
        self.metric_manager = dict()

        self.init_hardware_config()
        self.init_paths()
        if load_dataset:
            self.load_dataset()
        self.params["model_params"]["use_amp"] = self.params["training_params"]["use_amp"]
        self.output_file_suffix = ""

    def init_paths(self) -> None:
        """
        Create output folders for results and checkpoints
        """
        output_path = os.path.join("outputs", self.params["training_params"]["output_folder"])
        os.makedirs(output_path, exist_ok=True)
        checkpoints_path = os.path.join(output_path, "checkpoints")
        os.makedirs(checkpoints_path, exist_ok=True)
        results_path = os.path.join(output_path, "results")
        os.makedirs(results_path, exist_ok=True)

        self.paths = {
            "results": results_path,
            "checkpoints": checkpoints_path,
            "output_folder": output_path
        }

    def load_dataset(self) -> None:
        """
        Load datasets, data samplers and data loaders
        """
        for key in ["use_ddp", "train_batch_size", "val_batch_size", "test_batch_size"]:
            if key in self.params["training_params"]:
                self.params["dataset_params"][key] = self.params["training_params"][key]
        if "token_win" in self.params["model_params"]:
            self.params["dataset_params"]["token_win"] = self.params["model_params"]["token_win"]
        if "num_token_heads" in self.params["model_params"]:
            self.params["dataset_params"]["num_token_heads"] = self.params["model_params"]["num_token_heads"]

        self.params["dataset_params"]["num_gpu"] = len(self.params["training_params"]["gpus"]) if self.device != "cpu" else 0
        self.params["dataset_params"]["worker_per_gpu"] = 4 if "worker_per_gpu" not in self.params["dataset_params"] else self.params["dataset_params"]["worker_per_gpu"]
        self.dataset = self.params["dataset_params"]["dataset_manager"](self.params["dataset_params"])
        self.dataset.load_datasets()
        self.dataset.load_samplers()
        self.dataset.load_dataloaders()
        self.tokenizer = self.dataset.tokenizer

    def init_hardware_config(self) -> None:
        """
        Check available hardware resources
        """
        # Debug mode
        if self.params["training_params"]["force_cpu"] or not torch.cuda.is_available():
            self.params["training_params"]["use_ddp"] = False
            self.params["training_params"]["use_amp"] = False
            self.device = "cpu"
            print("WORKING ON CPU !\n")

        # Manage Distributed Data Parallel & GPU usage
        self.manual_seed = 1111 if "manual_seed" not in self.params["training_params"].keys() else self.params["training_params"]["manual_seed"]
        self.ddp_config = {
            "master": self.params["training_params"]["use_ddp"] and self.params["training_params"]["ddp_rank"] == 0,
            "address": "localhost" if "ddp_addr" not in self.params["training_params"].keys() else self.params["training_params"]["ddp_addr"],
            "port": "11111" if "ddp_port" not in self.params["training_params"].keys() else self.params["training_params"]["ddp_port"],
            "backend": "nccl" if "ddp_backend" not in self.params["training_params"].keys() else self.params["training_params"]["ddp_backend"],
            "rank": self.params["training_params"]["ddp_rank"],
        }
        self.is_master = self.ddp_config["master"] or not self.params["training_params"]["use_ddp"]

        if self.device != "cpu":
            self.autocast_mode = "cuda"
            if self.params["training_params"]["use_ddp"]:
                self.device = torch.device(self.ddp_config["rank"])
                self.params["dataset_params"]["ddp_rank"] = self.ddp_config["rank"]
                self.launch_ddp()
            else:
                self.device = torch.device("cuda:0")
            # Print GPU info
            # global
            if (self.params["training_params"]["use_ddp"] and self.ddp_config["master"]) or not self.params["training_params"]["use_ddp"]:
                print("##################")
                num_gpu = torch.cuda.device_count()
                print("Available GPUS: {}".format(num_gpu))
                self.params["training_params"]["nb_gpu"] = num_gpu
                for i in range(num_gpu):
                    print("Rank {}: {} {}".format(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i)))
                print("##################")
            # local
            print("Local GPU:")
            print("Rank {}: {} {}".format(self.params["training_params"]["ddp_rank"],
                                              torch.cuda.get_device_name(),
                                              torch.cuda.get_device_properties(self.device)))
            print("##################")
        self.params["model_params"]["device"] = self.device if isinstance(self.device, str) else self.device.type

    def load_model(self, reset_optimizer: bool=False, strict: bool=True, checkpoint: str|None=None) -> None:
        """
        Load model weights from scratch or from checkpoints, and initilize optimizer
        :param reset_optimizer: whether to reset the optimizer state (reset all learning rates)
        :param strict: strict parameter for torch module load_state_dict 
        :param checkpoint: path of checkpoint to load 
        """
        checkpoint = self.get_checkpoint() if checkpoint is None else checkpoint
        if checkpoint and "charset" in checkpoint and "keep_charset_from_checkpoint" in self.params["model_params"] and self.params["model_params"] ["keep_charset_from_checkpoint"]:
            self.params["model_params"]["vocab_size"] = len(checkpoint["charset"])
        # Instantiate Model
        for model_name in self.params["model_params"]["models"].keys():
            self.models[model_name] = self.params["model_params"]["models"][model_name](self.params["model_params"])
            self.models[model_name].to(self.device)  # To GPU or CPU
            # make the model compatible with Distributed Data Parallel if used
            if self.params["training_params"]["use_ddp"]:
                self.models[model_name] = DDP(self.models[model_name], [self.ddp_config["rank"]])


        # Handle curriculum dropout
        if "dropout_scheduler" in self.params["model_params"]:
            func = self.params["model_params"]["dropout_scheduler"]["function"]
            T = self.params["model_params"]["dropout_scheduler"]["T"]
            self.dropout_scheduler = DropoutScheduler(self.models, func, T)
            if checkpoint:
                self.dropout_scheduler.step(checkpoint["step"])
            self.dropout_scheduler.update_dropout_rate()

        self.scaler = GradScaler(self.autocast_mode,
                                 init_scale=self.params["training_params"]["gradscaler_init_value"],
                                 enabled=self.params["training_params"]["use_amp"])

        # Check if checkpoint exists
        if checkpoint is not None:
            self.load_existing_model(checkpoint, strict=strict)
        else:
            self.init_new_model()

        self.load_optimizers(checkpoint, reset_optimizer=reset_optimizer)

        if self.is_master:
            print("LOADED EPOCH: {}\n".format(self.latest_epoch), flush=True)

    def get_checkpoint(self) -> dict|None:
        """
        Return checkpoint of current experiment folder if exist, return None otherwise
        """
        if self.params["training_params"]["load_epoch"] in ("best", "last"):
            file_path = find_file_by_keyword(self.paths["checkpoints"], self.params["training_params"]["load_epoch"])
            if file_path:
                    return torch.load(file_path, map_location=self.device)
        return None

    def load_existing_model(self, checkpoint: dict, strict: bool=True) -> None:
        """
        Load information and weights from previous training checkpoint
        """
        self.latest_epoch = checkpoint["epoch"]
        if "step" in checkpoint:
            self.latest_step = checkpoint["step"]
        if "sample" in checkpoint:
            self.latest_sample = checkpoint["sample"]
        self.best = checkpoint["best"]
        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "training_hist_tokens" in checkpoint:
            self.dataset.train_dataset.hist_tokens = checkpoint["training_hist_tokens"]
        # Load model weights from past training
        for model_name in self.models.keys():
            state_dict_name = "{}_state_dict".format(model_name)
            if "module" in list(checkpoint[state_dict_name].keys())[0]:
                checkpoint = convert_checkpoint_from_ddp(checkpoint, state_dict_name)

            ## for compatibility with older versions
            if model_name == "decoder":
                for key_name in ["pe_1d.pe", "pe_2d.pe", "pe_1d.div"]:
                    if key_name in self.models[model_name].state_dict() and key_name not in checkpoint[f"{model_name}_state_dict"]:
                        checkpoint[f"{model_name}_state_dict"][key_name] = self.models[model_name].state_dict()[key_name]

            # for compatibility with prvious convnext version:
            for key in ["bias", "weight"]:
                old_key = f"norm.{key}"
                if old_key in checkpoint[state_dict_name]:
                    del checkpoint[state_dict_name][old_key]
                    checkpoint.pop(f'optimizer_{model_name}_state_dict', None)
                                                      
            self.models[model_name].load_state_dict(checkpoint[state_dict_name], strict=strict)
        self.tokenizer = checkpoint["tokenizer"]

    def init_new_model(self) -> None:
        """
        Initialize model from scratch, apply transfer learning if specified in training configuration
        """
        # Specific weights initialization if exists
        for model_name in self.models.keys():
            try:
                self.models[model_name].init_weights()
            except:
                pass

        # Handle transfer learning instructions
        if self.params["model_params"]["transfer_learning"]:
            # Iterates over models
            for model_name in self.params["model_params"]["transfer_learning"].keys():
                state_dict_name, path, learnable, strict = self.params["model_params"]["transfer_learning"][model_name]
                path = check_exist_or_download_checkpoint(path)
                if os.path.isdir(path):
                    path = find_file_by_keyword(path, self.params["model_params"]["transfer_key"])
                # Loading pretrained weights file
                checkpoint = torch.load(path, map_location=self.device)
                state_dict_name = "{}_state_dict".format(state_dict_name)
                if not self.params["training_params"]["use_ddp"] and "module" in list(checkpoint[state_dict_name].keys())[0]:
                    checkpoint = convert_checkpoint_from_ddp(checkpoint, state_dict_name)
                if "convnext-v2" in path:
                    for key in ["bias", "weight"]:
                        old_key = f"norm.{key}"
                        new_key = f"final_norm.{key}"
                        if old_key in checkpoint[state_dict_name]:
                            if hasattr(self.models[model_name], "final_norm"):
                                checkpoint[state_dict_name][new_key] = checkpoint[state_dict_name][old_key]
                                del checkpoint[state_dict_name][old_key]
                            else:
                                del checkpoint[state_dict_name][old_key]

                print(f"Loaded weights for transfer learning: {path}")
                try:
                    # Load pretrained weights for model
                    self.models[model_name].load_state_dict(checkpoint[state_dict_name], strict=strict)
                    print("transfered weights for {}".format(state_dict_name), flush=True)
                except RuntimeError as e:
                    print(e, flush=True)
                    # if error, try to load each parts of the model (useful if only few layers are different)
                    for key in checkpoint[state_dict_name].keys():
                        try:
                            # for pre-training of decision layer 
                            if ("end_conv" in key or "emb" in key) and "transfered_charset" in self.params["model_params"] and self.params["model_params"]["transfered_charset"]:
                                self.adapt_decision_layer_to_old_charset(model_name, key, checkpoint, state_dict_name)
                            else:
                                self.models[model_name].load_state_dict(
                                    {key: checkpoint[state_dict_name][key]}, strict=False)
                        except RuntimeError as e:
                            ## exception when adding linebreak token from pretraining
                                print(e, flush=True)
                # Set parameters no trainable
                if not learnable:
                    self.set_model_learnable(self.models[model_name], False)

    def adapt_decision_layer_to_old_charset(self, model_name: str, key: str, checkpoint: dict, state_dict_name: str) -> None:
        """
        Transfer learning of the decision layer in case of close charsets between pre-training and training
        Useful when going from CTC pre-training to attention-based training
        """
        pretrained_chars = list()
        weights = checkpoint[state_dict_name][key]
        # new_size = list(weights.size())
        # new_size[0] = self.tokenizer.num_tokens + self.params["model_params"]["additional_tokens"]
        new_size = dict(self.models[model_name].named_parameters())[key].size()
        new_weights = torch.zeros(new_size, device=weights.device, dtype=weights.dtype)
        old_charset = checkpoint["tokenizer"].tokens_list
        if "bias" not in key:
            kaiming_uniform_(new_weights, nonlinearity="relu")
        for i, c in enumerate(self.tokenizer.charset):
            if c in old_charset:
                new_weights[i] = weights[old_charset.index(c)]
                if type(self.models[model_name]).__name__ in ["MTDANDecoder", "MTFasterDANDecoder"] and "end_conv" in key:
                    if checkpoint['params']['model_params']['models']['decoder'].__name__ in ["MTDANDecoder", "MTFasterDANDecoder"]:
                        old_num_heads = checkpoint['params']['model_params']['num_token_heads']
                        old_voc_size = weights.size(0) // old_num_heads
                    else:
                        old_num_heads = 1
                        old_voc_size = weights.size(0)
                    num_heads = self.models[model_name].num_token_heads
                    voc_size = new_weights.size(0) // num_heads
                    for k in range(1, num_heads):
                        new_weights[k*voc_size+i] = weights[k*old_voc_size+old_charset.index(c)] if k < old_num_heads else weights[old_charset.index(c)]
                pretrained_chars.append(c)
        if "transfered_charset_last_is_ctc_blank" in self.params["model_params"] and self.params["model_params"]["transfered_charset_last_is_ctc_blank"]:
            new_weights[-1] = weights[-1]
            pretrained_chars.append("<blank>")
        checkpoint[state_dict_name][key] = new_weights
        self.models[model_name].load_state_dict({key: checkpoint[state_dict_name][key]}, strict=False)
        print("Pretrained chars for {} ({}): {}".format(key, len(pretrained_chars), pretrained_chars))
 
    def load_optimizers(self, checkpoint: dict, reset_optimizer: bool=False) -> None:
        """
        Load the optimizer for each model (encoder/decoder)
        """
        for model_name in self.models.keys():
            new_params = dict()
            if checkpoint and "optimizer_named_params_{}".format(model_name) in checkpoint:
                self.optimizers_named_params_by_group[model_name] = checkpoint["optimizer_named_params_{}".format(model_name)]
                # for progressively growing models
                for name, param in self.models[model_name].named_parameters():
                    existing = False
                    for gr in self.optimizers_named_params_by_group[model_name]:
                        if name in gr:
                            gr[name] = param
                            existing = True
                            break
                    if not existing:
                        new_params.update({name: param})
            else:
                self.optimizers_named_params_by_group[model_name] = [dict(), ]
                self.optimizers_named_params_by_group[model_name][0].update(self.models[model_name].named_parameters())

            # Instantiate optimizer
            self.reset_optimizer(model_name)

            # Handle learning rate schedulers
            if "lr_schedulers" in self.params["training_params"] and self.params["training_params"]["lr_schedulers"]:
                key = "all" if "all" in self.params["training_params"]["lr_schedulers"] else model_name
                if key in self.params["training_params"]["lr_schedulers"]:
                    self.lr_schedulers[model_name] = self.params["training_params"]["lr_schedulers"][key]["class"]\
                        (self.optimizers[model_name], **self.params["training_params"]["lr_schedulers"][key]["args"])

            # Load optimizer state from past training
            if checkpoint and not reset_optimizer and "optimizer_{}_state_dict".format(model_name) in checkpoint:
                self.optimizers[model_name].load_state_dict(checkpoint["optimizer_{}_state_dict".format(model_name)])
                # Load optimizer scheduler config from past training if used
                if "lr_schedulers" in self.params["training_params"] and self.params["training_params"]["lr_schedulers"] \
                        and "lr_scheduler_{}_state_dict".format(model_name) in checkpoint.keys():
                    self.lr_schedulers[model_name].load_state_dict(checkpoint["lr_scheduler_{}_state_dict".format(model_name)])

            # for progressively growing models, keeping learning rate
            if checkpoint and new_params:
                self.optimizers_named_params_by_group[model_name].append(new_params)
                self.optimizers[model_name].add_param_group({"params": list(new_params.values())})

    @staticmethod
    def set_model_learnable(model: Module, learnable: bool =True) -> None:
        """
        Modify the learnability of module weights (except last two layers)
        """
        for k, p in list(model.named_parameters())[:-2]:  # except last depth sep conv for fcn encoder
            p.requires_grad = learnable

    @staticmethod
    def is_model_learnable(model: Module) -> bool:
        """
        Check if at least one parameter is learnable in given torch module
        """
        for p in list(model.parameters()):
            if p.requires_grad:
                return True
        return False

    def save_model(self, n: int, name: str, keep_weights=False) -> None:
        """
        Save model weights and training info for curriculum learning or learning rate for instance
        :param n: integer (epoch/sample number)
        :param name: best/last
        :param keep_weights: whether to keep old weights or not. If False, only keep last weights, and best weights on val set.
        """
        if name == "best":
            eval_mode = self.params["training_params"]["eval_on_valid"]["mode"]
        else:
            eval_mode = self.params["training_params"]["save_weights"]["mode"]
        if not self.is_master:
            return
        to_del = []
        for filename in os.listdir(self.paths["checkpoints"]):
            if name in filename:
                to_del.append(os.path.join(self.paths["checkpoints"], filename))
        path = os.path.join(self.paths["checkpoints"], "{}_{}_{}.pt".format(name, eval_mode, n))
        content = {
            'optimizers_named_params': self.optimizers_named_params_by_group,
            'epoch': self.latest_epoch,
            'step': self.latest_step,
            'sample': self.latest_sample,
            "scaler_state_dict": self.scaler.state_dict(),
            'best': self.best,
            "tokenizer": self.tokenizer,
            "params": self.params,
        }
        for model_name in self.optimizers:
            content['optimizer_{}_state_dict'.format(model_name)] = self.optimizers[model_name].state_dict()
        for model_name in self.lr_schedulers:
            content["lr_scheduler_{}_state_dict".format(model_name)] = self.lr_schedulers[model_name].state_dict()
        for model_name in self.models.keys():
            content["{}_state_dict".format(model_name)] = self.models[model_name].state_dict()
        if len(self.dataset.train_dataset.hist_tokens) > 0:
            content["training_hist_tokens"] = self.dataset.train_dataset.hist_tokens
        torch.save(content, path)
        if not keep_weights:
            for path_to_del in to_del:
                if path_to_del != path:
                    os.remove(path_to_del)

    def reset_optimizers(self) -> None:
        """
        Reset learning rate of all optimizers
        """
        print("Resetting optimizers")
        for model_name in self.models.keys():
            self.reset_optimizer(model_name)

    def reset_optimizer(self, model_name: str) -> None:
        """
        Reset optimizer learning rate for given model name
        """
        params = list(self.optimizers_named_params_by_group[model_name][0].values())
        key = "all" if "all" in self.params["training_params"]["optimizers"] else model_name
        self.optimizers[model_name] = self.params["training_params"]["optimizers"][key]["class"](params, **self.params["training_params"]["optimizers"][key]["args"])
        for i in range(1, len(self.optimizers_named_params_by_group[model_name])):
            self.optimizers[model_name].add_param_group({"params": list(self.optimizers_named_params_by_group[model_name][i].values())})

    def save_params(self) -> None:
        """
        Output json file containing a summary of all hyperparameters chosen for the training
        """
        def compute_nb_params(module):
            return sum([np.prod(p.size()) for p in list(module.parameters())])

        def class_to_str_dict(my_dict):
            for key in my_dict.keys():
                if callable(my_dict[key]):
                    my_dict[key] = my_dict[key].__name__
                elif isinstance(my_dict[key], np.ndarray):
                    my_dict[key] = my_dict[key].tolist()
                elif isinstance(my_dict[key], dict):
                    my_dict[key] = class_to_str_dict(my_dict[key])
            return my_dict

        shutil.copy(os.path.join("configs/", self.params["args"]["config"]), os.path.join(self.paths["results"], self.params["args"]["config"]))
        path = os.path.join(self.paths["results"], "params.txt")
        i = 1
        while os.path.isfile(path):
            path = "{}_{}.txt".format(os.path.join(self.paths["results"], "params"), i)
            i += 1
        params = copy.deepcopy(self.params)
        params = class_to_str_dict(params)
        params["date"] = date.today().strftime("%d/%m/%Y")
        total_params = 0
        for model_name in self.models.keys():
            current_params = compute_nb_params(self.models[model_name])
            params["model_params"]["models"][model_name] = [params["model_params"]["models"][model_name], "{:,}".format(current_params)]
            total_params += current_params
        params["model_params"]["total_params"] = "{:,}".format(total_params)

        params["hardware"] = dict()
        if self.device != "cpu":
            for i in range(self.params["training_params"]["nb_gpu"]):
                params["hardware"][str(i)] = "{} {}".format(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i))
        else:
            params["hardware"]["0"] = "CPU"
        params["software"] = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)

    def update_memory_consumption(self) -> None:
        """
        Log information about memory consumption.
        """
        self.max_mem_usage_by_epoch.append(torch.cuda.max_memory_allocated())
        if self.device != "cpu":
            torch.cuda.reset_max_memory_allocated()
        with open(os.path.join(self.paths["results"], "memory.txt"), 'a') as f:
            current = round(self.max_mem_usage_by_epoch[-1]/1e9, 2)
            max = round(np.max(self.max_mem_usage_by_epoch)/1e9, 2)
            min = round(np.min(self.max_mem_usage_by_epoch)/1e9, 2)
            median = round(np.median(self.max_mem_usage_by_epoch)/1e9, 2)
            mean = round(np.mean(self.max_mem_usage_by_epoch)/1e9, 2)
            f.write("E{} - Current: {} Go - Max: {} Go - Min: {} Go - Mean: {} Go - Median: {} Go\n".format(
                self.latest_epoch, current, max, min, mean, median))

    def backward_loss(self, loss: Tensor, retain_graph: bool=False) -> None:
        """
        Encapsulation of torch loss backward
        """
        self.scaler.scale(loss).backward(retain_graph=retain_graph)

    def step_optimizers(self, names: list|None=None) -> None:
        """
        Encapsulation of torch step optimizer.
        Perform a step for all models if names is None, or only those given in names otherwise.
        """
        if self.params["training_params"]["log_gradients"]:
            self.log_gradients()
        self.grad_acc_ind += 1
        if self.grad_acc_ind >= self.grad_acc_step:
            for model_name in self.optimizers:
                if names and model_name not in names:
                    continue
                if not self.is_model_learnable(self.models[model_name]):
                    continue
                if "gradient_clipping" in self.params["training_params"] and model_name in self.params["training_params"]["gradient_clipping"]["models"]:
                    self.scaler.unscale_(self.optimizers[model_name])
                    torch.nn.utils.clip_grad_norm_(self.models[model_name].parameters(), self.params["training_params"]["gradient_clipping"]["max"])
                self.scaler.step(self.optimizers[model_name])
            self.scaler.update()
            self.latest_step += 1
            self.zero_optimizers()
            self.grad_acc_ind = 0

    def zero_optimizers(self, set_to_none: bool=True) -> None:
        """
        Zero gradients for all optimizers
        """
        for model_name in self.optimizers:
            self.zero_optimizer(model_name, set_to_none)

    def zero_optimizer(self, model_name: str, set_to_none: bool=True) -> None:
        """
        Zero gradients for optimizer of given model_name
        """
        self.optimizers[model_name].zero_grad(set_to_none=set_to_none)

    def train(self) -> None:
        """
        Main training loop
        """
        # init tensorboard file and output param summary file
        if self.is_master:
            self.writer = SummaryWriter(self.paths["results"])
            self.save_params()
        # init variables
        self.begin_time = time()
        nb_epochs = self.params["training_params"]["max_nb_epochs"]
        metric_names = self.params["training_params"]["train_metrics"]
        if "gradient_accumulation" in self.params["training_params"]:
            self.grad_acc_step = self.params["training_params"]["gradient_accumulation"]
        max_train_samples = self.params["training_params"]["max_num_samples"] if "max_num_samples" in self.params["training_params"] else None
        batch_size = self.params["training_params"]["train_batch_size"]
        eval_config = self.params["training_params"]["eval_on_valid"] if isinstance(self.params["training_params"]["eval_on_valid"], dict) else None
        save_config = self.params["training_params"]["save_weights"]
        reset_optimizer_num_samples = self.params["training_params"]["reset_optimizer_after_k_samples"] if "reset_optimizer_after_k_samples" in self.params["training_params"] else np.inf
        stop_training = False
        display_values = None
        # perform epochs
        for num_epoch in range(self.latest_epoch+1, nb_epochs):
            self.dataset.train_dataset.training_info.update({
                "epoch": self.latest_epoch,
                "step": self.latest_step,
                "num_samples": self.latest_sample
            })
            self.phase = "train"
            # Check maximum training time stop condition
            if stop_training or self.max_training_time and time() - self.begin_time > self.max_training_time:
                break
            # set models trainable
            for model_name in self.models.keys():
                self.models[model_name].train()
            self.latest_epoch = num_epoch
            if self.dataset.train_dataset.curriculum_config:
                self.dataset.train_dataset.curriculum_config["epoch"] = self.latest_epoch
            # init epoch metrics values
            self.metric_manager["train"] = MetricManager(metric_names=metric_names)

            with tqdm(total=len(self.dataset.train_loader.dataset), disable=not self.is_master) as pbar:
                pbar.set_description("EPOCH {}/{}".format(num_epoch, nb_epochs))
                # iterates over mini-batch data
                for ind_batch, batch_data in enumerate(self.dataset.train_loader):
                    self.latest_batch = ind_batch + 1
                    self.total_batch += 1
                    self.dataset.train_dataset.training_info["num_samples"] = self.latest_sample
                    self.dataset.train_dataset.training_info["step"] = self.latest_step

                    # Check max samples
                    if max_train_samples is not None and self.latest_sample > max_train_samples:
                        stop_training = True
                        break

                    # train on batch data and compute metrics
                    batch_values = self.train_batch(batch_data, metric_names)
                    batch_metrics = self.metric_manager["train"].compute_metrics(batch_values, metric_names)
                    batch_metrics["names"] = batch_data["names"]
                    batch_metrics["ids"] = batch_data["ids"]
                    # Merge metrics if Distributed Data Parallel is used
                    if self.params["training_params"]["use_ddp"]:
                        batch_metrics = self.merge_ddp_metrics(batch_metrics)
                    # Update learning rate via scheduler if one is used
                    if "lr_schedulers" in self.params["training_params"] and self.params["training_params"]["lr_schedulers"]:
                        for model_name in self.models:
                            key = "all" if "all" in self.params["training_params"]["lr_schedulers"] else model_name
                            if model_name in self.lr_schedulers and ind_batch % self.params["training_params"]["lr_schedulers"][key]["step_interval"] == 0:
                                self.lr_schedulers[model_name].step(len(batch_metrics["names"]))
                                if "lr" in metric_names:
                                    self.writer.add_scalar("lr_{}".format(model_name), self.lr_schedulers[model_name].lr, self.lr_schedulers[model_name].step_num)
                    # Update dropout scheduler if used
                    if self.dropout_scheduler:
                        self.dropout_scheduler.step(len(batch_metrics["names"]))
                        self.dropout_scheduler.update_dropout_rate()

                    # Add batch metrics values to epoch metrics values
                    self.metric_manager["train"].update_metrics(batch_metrics)
                    display_values = self.metric_manager["train"].get_display_values()
                    pbar.set_postfix(values=str(display_values))

                    num_batch_samples = len(batch_metrics["names"])
                    pbar.update(num_batch_samples)
                    self.latest_sample += num_batch_samples

                    # Check eval after k samples
                    if eval_config and eval_config["mode"] == "sample" and self.latest_sample >= eval_config["start_at"] and self.latest_sample % eval_config["interval"] < batch_size:
                        self.eval_on_val(self.latest_sample)

                    # Check eval after k backprop steps
                    if eval_config and eval_config["mode"] == "step" and self.latest_step >= eval_config["start_at"] and self.latest_step % eval_config["interval"] == 0:
                        self.eval_on_val(self.latest_step)

                    if save_config["mode"] == "sample" and self.latest_sample % save_config["interval"] < batch_size:
                        self.save_last_weights(self.latest_sample)

                    if 2*reset_optimizer_num_samples > self.latest_sample >= reset_optimizer_num_samples and self.latest_sample % reset_optimizer_num_samples < batch_size:
                        self.reset_optimizers()

            # log metrics in tensorboard file
            if self.is_master:
                for key in display_values.keys():
                    self.writer.add_scalar('{}_{}'.format("train", key), display_values[key], num_epoch)
            self.latest_train_metrics = display_values

            if eval_config and eval_config["mode"] == "epoch" and self.latest_epoch >= eval_config["start_at"] and num_epoch % eval_config["interval"] == 0:
                self.eval_on_val(num_epoch)

            if save_config["mode"] == "epoch" and self.latest_epoch % save_config["interval"] == 0:
                self.save_last_weights(self.latest_epoch)

    def eval_on_val(self, n: int) -> None:
        """
        Perform evaluation of all val sets specified in training configuration.
        :param n: number of epoch/sample (for checkpoint name).
        """
        focus_metric_name = self.params["training_params"]["focus_metric"]
        for valid_set_name in self.dataset.valid_loaders.keys():
            # evaluate set and compute metrics
            eval_values = self.evaluate(valid_set_name, split_name="val")
            self.latest_valid_metrics[valid_set_name] = eval_values
            # log valid metrics in tensorboard file
            if self.is_master:
                for key in eval_values.keys():
                    self.writer.add_scalar('{}_{}'.format(valid_set_name, key), eval_values[key], n)
                if (valid_set_name == self.params["training_params"]["set_name_focus_metric"] and
                        (self.best is None or (eval_values[focus_metric_name] <= self.best and
                                               self.params["training_params"]["expected_metric_value"] == "low") or
                         (eval_values[focus_metric_name] >= self.best and self.params["training_params"]["expected_metric_value"] == "high"))):
                    self.best = eval_values[focus_metric_name]
                    self.save_model(n=n, name="best")

        if self.is_master and self.params["training_params"]["set_name_focus_metric"] == "all-val":
            value = np.mean([values[focus_metric_name] for values in self.latest_valid_metrics.values()]) 
            self.writer.add_scalar('{}_{}'.format( "all-val", focus_metric_name), value, n)
            if self.best is None or (value <= self.best and self.params["training_params"]["expected_metric_value"] == "low") or \
                         (value >= self.best and self.params["training_params"]["expected_metric_value"] == "high"):
                self.best = value
                self.save_model(n=n, name="best")

    def save_last_weights(self, n: int) -> None:
        """
        Checkpoint model as weights of last trained session.
        :param n: number of current epoch/samples. 
        """
        if self.is_master:
            self.save_model(n=n, name="last")
            self.update_memory_consumption()
            if self.params["training_params"]["save_weights"]["keep_all"]:
                self.save_model(n=n, name="weights", keep_weights=True)
            self.writer.flush()

    def get_output_file_path(self, set_name: str) -> str:
        """
        Create filepath for current prediction metrics.
        """
        filename = f"predict_{set_name}_e{self.latest_epoch}_b{self.params['training_params']['test_batch_size']}{self.output_file_suffix}.json"
        return os.path.join(self.paths["results"], filename)

    @staticmethod
    def np_encoder(object):
        """
        Useful for json export.
        """
        if isinstance(object, np.generic):
            return object.item()

    def evaluate(self, set_name: str, split_name: str, output: bool=False) -> None:
        """
        Main loop for evaluation.
        :param set_name: name of dataset to evaluate (ex: rimes2009_page-val)
        :param split_name: "train", "val" or "test"
        :param output: whether to output computed metrics in a json file. 
        """
        self.phase = "eval"
        if split_name == "val":
            loader = self.dataset.valid_loaders[set_name]
        else:
            if set_name not in self.dataset.test_loaders:
                self.dataset.generate_test_loader(set_name)
            loader = self.dataset.test_loaders[set_name]
        batch_size = loader.batch_size

        # Set models in eval mode
        for model_name in self.models.keys():
            self.models[model_name].eval()
        metric_names = self.params["training_params"]["eval_metrics"]
        display_values = None

        # initialize epoch metrics
        self.metric_manager[set_name] = MetricManager(metric_names)
        with tqdm(total=len(loader.dataset), disable=not self.is_master) as pbar:
            pbar.set_description("Evaluation on {} E{}".format(set_name, self.latest_epoch))
            with torch.no_grad():
                # iterate over batch data
                for ind_batch, batch_data in enumerate(loader):
                    self.latest_batch = ind_batch + 1
                    # eval batch data and compute metrics
                    batch_values = self.evaluate_batch(batch_data, metric_names)
                    batch_metrics = self.metric_manager[set_name].compute_metrics(batch_values, metric_names)
                    batch_metrics["names"] = batch_data["names"]
                    batch_metrics["ids"] = batch_data["ids"]
                    # merge metrics values if Distributed Data Parallel is used
                    if self.params["training_params"]["use_ddp"]:
                        batch_metrics = self.merge_ddp_metrics(batch_metrics)

                    # add batch metrics to epoch metrics
                    self.metric_manager[set_name].update_metrics(batch_metrics)
                    display_values = self.metric_manager[set_name].get_display_values()

                    pbar.set_postfix(values=str(display_values))
                    pbar.update(len(batch_metrics["names"]))

        if output:
            if "pred" in metric_names:
                self.output_pred(set_name, batch_size)
            metrics = self.metric_manager[set_name].get_display_values(output=True)
            path = self.get_output_file_path(set_name)
            with open(path, "w") as f:
                if "worse_cer" in metric_names:
                    list_cers = np.array(self.metric_manager[set_name].get("edit_chars")) / np.array(
                        self.metric_manager[set_name].get("nb_chars"))
                    list_edits = np.array(self.metric_manager[set_name].get("edit_chars"))
                    list_names = self.metric_manager[set_name].get("names")
                    indices_cers = np.argsort(list_cers)[-10:][::-1]
                    indices_edits = np.argsort(list_edits)[-10:][::-1]
                    worse_cer_list = ["{}% - {}\n".format(np.round(list_cers[ind] * 100, 2), list_names[ind]) for ind in indices_cers]
                    worse_edit_distance = ["{} - {}\n".format(list_edits[ind], list_names[ind]) for ind in indices_edits]
                    metrics["worse_cer"] = worse_cer_list
                    metrics["worse_edit_distance"] = worse_edit_distance
                # if "best_cer" in metric_names:
                #     list_cers = np.array(self.metric_manager[set_name].get("edit_chars")) / np.array(
                #         self.metric_manager[set_name].get("nb_chars"))
                #     list_edits = np.array(self.metric_manager[set_name].get("edit_chars"))
                #     list_names = self.metric_manager[set_name].get("names")
                #     indices_cers = np.argsort(list_cers)[:10]
                #     indices_edits = np.argsort(list_edits)[:10]
                #     f.write("Best CER:\n")
                #     for ind in indices_cers:
                #         f.write("{}% - {}\n".format(np.round(list_cers[ind] * 100, 2), list_names[ind]))
                #     f.write("Best Edit CER:\n")
                #     for ind in indices_edits:
                #         f.write("{} - {}\n".format(list_edits[ind], list_names[ind]))
                # if "maurdor_details" in metric_names:
                #     self.output_cer_details_maurdor(set_name, f)
                json.dump(metrics, f, indent=4, default=self.np_encoder)

        if split_name == "test":
            self.dataset.remove_test_dataset(set_name)
        return display_values


    def output_pred(self, name:str, batch_size: int) -> None:
        """
        Output raw text prediction from current metric manager as txt file.
        :param name: key of metric_manager in which to look for predictions.
        :param batch_size: size of mini-batch used (for file name).
        """
        path = os.path.join(self.paths["results"], "pred_{}_{}_b{}.txt".format(name, self.latest_epoch, batch_size))
        pred = "\n".join(self.metric_manager[name].get("pred"))
        with open(path, "w") as f:
            f.write(pred)


    def launch_ddp(self) -> None:
        """
        Initialize Distributed Data Parallel system
        """
        mp.set_start_method('fork', force=True)
        os.environ['MASTER_ADDR'] = self.ddp_config["address"]
        os.environ['MASTER_PORT'] = str(self.ddp_config["port"])
        dist.init_process_group(self.ddp_config["backend"], rank=self.ddp_config["rank"], world_size=len(self.params["training_params"]["gpus"]))
        torch.cuda.set_device(self.ddp_config["rank"])
        random.seed(self.manual_seed)
        np.random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed(self.manual_seed)

    def merge_ddp_metrics(self, metrics):
        """
        Merge metrics when Distributed Data Parallel is used
        """
        outputs = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(outputs, metrics)
        merged_outputs = dict()
        for metric_name in metrics.keys():
            merged_outputs[metric_name] = list(itertools.chain(*[o[metric_name] for o in outputs]))
        return merged_outputs

    @staticmethod
    def cleanup():
        """
        Clean after DDP usage.
        """
        dist.destroy_process_group()

    def train_batch(self, batch_data, metric_names):
        raise NotImplementedError

    def evaluate_batch(self, batch_data, metric_names):
        raise NotImplementedError

    def predict_batch(self, batch_data):
        raise NotImplementedError

    def log_gradients(self) -> None:
        """
        Log gradients info in tensorboard logging file.
        """
        grads = list()
        for model_name in self.models.keys():
            for name, param in self.models[model_name].named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    if torch.any(torch.isnan(grad)):
                        print(f"Step: {self.latest_step} - Nan gradient in {name} params")
                    if torch.any(torch.isinf(grad)):
                        print(f"Step: {self.latest_step} - Inf gradient in {name} params")
                    grads.append(grad)
                # self.writer.add_scalar(f"{name}_grad_abs_median", torch.median(torch.abs(grad)).item() , self.latest_step)
        self.writer.add_scalar("gradient_min", min([torch.min(g).item() for g in grads]), self.latest_step)
        self.writer.add_scalar("gradient_max", max([torch.max(g).item() for g in grads]), self.latest_step)
        self.writer.add_scalar("gradient_norm_mean", torch.mean(torch.stack([g.norm() for g in grads])).item(), self.latest_step)
        self.writer.add_scalar("gradient_norm_median", torch.median(torch.stack([g.norm() for g in grads])).item(), self.latest_step)
        self.writer.add_scalar("gradient_norm_std", torch.std(torch.stack([g.norm() for g in grads])).item(), self.latest_step)

