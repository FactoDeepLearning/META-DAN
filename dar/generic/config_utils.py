"""
Module handling preprocessing of parameters from config file/argparse.
"""

from torch.optim import Adam
from dar.generic.transforms import line_aug_config, aug_config
from dar.generic.scheduler import exponential_dropout_scheduler
import yaml
import os


DEFAULT_CONFIG = {
    "line_ctc": "line_ctc_template.yaml",
    "dan": "dan_template_page.yaml",
    "faster_dan": "faster_dan_template_page.yaml",
    "mt_dan": "mt_dan_template_page.yaml",
    "w_dan": "w_dan_template_page.yaml",
    "meta_dan": "meta_dan_template_page.yaml",
    "span": "span_template_page.yaml",
    "van": "van_template_page.yaml",
}


def load_yaml_file(path: str) -> dict:
    """
    Load a yaml file from path as a dict
    """
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def preprocess_params(params: dict) -> dict:
    """
    Preprocess the param dict to adapt it to run script behavior
    """
    params = preprocess_params_dataset(params)
    params = preprocess_params_synthetic_config(params)
    params = preprocess_params_optimizer(params)
    params = preprocess_params_data_augmentation(params)
    params = preprocess_params_dropout_scheduler(params)
    return params

def preprocess_params_dataset(params: dict) -> dict:
    """
    Preprocess parameters related to dataset configuration.
    """
    args = params["args"]

    if args["dataset"]:
        params["dataset_params"].update({
            "train": [{"dataset_name": args["dataset"], "split": "train"}],
            "val": [{
                "name": f"{args['dataset'] }-val",
                "sets": [{"dataset_name": args["dataset"], "split": "val"}],
            }],
            "test": [{
                "name": f"{args['dataset']}-test",
                "sets": [{"dataset_name": args["dataset"], "split": "test"}],
            }]
        })
        params["training_params"]["set_name_focus_metric"] = f"{args['dataset'] }-val"

    if args["dataset_config"] is not None:
        yaml_path = os.path.join("configs", "data_configs", f"{args['dataset_config']}.yaml")
        data_config = load_yaml_file(yaml_path)

        params["dataset_params"].update({
            "train": data_config["datasets"],
            "val": [{
                "name": f"{dataset['dataset_name']}-val",
                "sets": [{"dataset_name": dataset['dataset_name'], "split": "val"}],
            } for dataset in data_config["datasets"] if "no_val" not in dataset or not dataset["no_val"]],
            "test": [{
                "name": f"{dataset['dataset_name']}-test",
                "sets": [{"dataset_name": dataset['dataset_name'], "split": "test"}],
            } for dataset in data_config["datasets"] if "no_test" not in dataset or not dataset["no_test"]]
        })
        for train_config in params["dataset_params"]["train"]:
            train_config.update({"split": "train"})
        params["training_params"]["set_name_focus_metric"] = "all-val" 
    return params


def preprocess_params_optimizer(params: dict) -> dict:
    """
    Preprocess parameters related to optimizer configuration.
    """
    for key in params["training_params"]["optimizers"]:
        if isinstance(params["training_params"]["optimizers"][key], dict):
            for subkey in params["training_params"]["optimizers"][key]:
                if subkey == "class":
                    if params["training_params"]["optimizers"][key][subkey] == "Adam":
                        params["training_params"]["optimizers"][key][subkey] = Adam
    return params


def preprocess_params_data_augmentation(params: dict) -> dict:
    """
    Preprocess parameters related to the data augmentation strategy.
    """
    if "data_augmentation" in params["dataset_params"]["config"] and params["dataset_params"]["config"]["data_augmentation"]:
        augmentation_func = params["dataset_params"]["config"]["data_augmentation"]
        if augmentation_func == "text_line_aug":
            params["dataset_params"]["config"]["data_augmentation"] = line_aug_config(0.9, 0.1)
        if augmentation_func == "text_doc_aug":
            params["dataset_params"]["config"]["data_augmentation"] = aug_config(0.9, 0.1)
    return params


def preprocess_params_dropout_scheduler(params: dict) -> dict:
    """
    Preprocess parameters related to the dropout scheduling strategy.
    """
    if "dropout_scheduler" in params["model_params"] and params["model_params"]["dropout_scheduler"]:
        if params["model_params"]["dropout_scheduler"]["function"] == "exponential":
            params["model_params"]["dropout_scheduler"]["function"] = exponential_dropout_scheduler
    return params


def preprocess_params_synthetic_config(params: dict) -> dict:
    """
    Preprocess the parameters related to the synthetic data generation.
    """
    if "synthetic_data" in params["dataset_params"]["config"] and params["dataset_params"]["config"]["synthetic_data"] is not None:
        config_file_path = os.path.join("configs", "syn_configs", params["dataset_params"]["config"]["synthetic_data"])
        params["dataset_params"]["config"]["synthetic_data"] = load_yaml_file(config_file_path)
        syn_config = params["dataset_params"]["config"]["synthetic_data"]
        params["training_params"]["train_metrics"].append("syn_max_lines")
        params["training_params"]["train_metrics"].append("syn_ratio")
        if "syn_mode" in params["dataset_params"]["config"]:
            syn_config["mode"] = params["dataset_params"]["config"]["syn_mode"]
        if params["args"]["dataset_config"] is not None and 'syn_config' in params["dataset_params"]["train"][0]:
            syn_config["custom_config"] = {dataset['dataset_name']: load_yaml_file(os.path.join("configs", "syn_generator_configs", dataset['syn_config'])) for dataset in params["dataset_params"]["train"]}
        elif "syn_custom_config" in params["dataset_params"]["config"]:
            syn_config_file_path = os.path.join("configs", "syn_generator_configs", params["dataset_params"]["config"]["syn_custom_config"]+".yaml")
            syn_config["custom_config"] = {dataset['dataset_name']: load_yaml_file(syn_config_file_path) for dataset in params["dataset_params"]["train"]}
        elif "custom_config" in syn_config:
            syn_config["custom_config"] = {dataset['dataset_name']: load_yaml_file(os.path.join("configs", "syn_generator_configs", syn_config["custom_config"]+".yaml")) for dataset in params["dataset_params"]["train"]}
        if "syn_source" in params["dataset_params"]["config"]:
            syn_config["text_source"] = params["dataset_params"]["config"]["syn_source"]
        if "syn_init_proba" in params["dataset_params"]["config"]:
           syn_config["scheduler"]["init_proba"] = params["dataset_params"]["config"]["syn_init_proba"]
        if "syn_end_proba" in params["dataset_params"]["config"]:
            syn_config["scheduler"]["end_proba"] = params["dataset_params"]["config"]["syn_end_proba"]
        if "syn_num_samples" in params["dataset_params"]["config"]:
            syn_config["scheduler"]["num_samples_proba"] = params["dataset_params"]["config"]["syn_num_samples"]
        if "syn_double_line" in params["dataset_params"]["config"]:
            syn_config["syn_double_line"] = params["dataset_params"]["config"]["syn_double_line"]
        if "add_loop_syn" in params["dataset_params"]["config"]:
            syn_config["add_loop_syn"] = params["dataset_params"]["config"]["add_loop_syn"]
        if "syn_single_column" in params["dataset_params"]["config"]:
            syn_config["syn_single_column"] = params["dataset_params"]["config"]["syn_single_column"]
        if "curriculum" in syn_config and syn_config["curriculum"] is not None:
            if "curr_num_samples" in params["dataset_params"]["config"]:
                syn_config["curriculum"]["num_samples_curr"] = params["dataset_params"]["config"]["curr_num_samples"]
            if "start_curr_at_sample" in params["dataset_params"]["config"]:
                syn_config["start_curr_at_sample"] = params["dataset_params"]["config"]["start_curr_at_sample"]

    return params