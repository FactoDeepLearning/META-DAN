"""
Module handling all stuff related to metrics
"""

import re
import editdistance
import numpy as np
from dar.dataset_format.read2016 import get_sem_tokens as get_sem_tokens_read2016
from dar.dataset_format.rimes2009 import get_sem_tokens as get_sem_tokens_rimes2009
from scipy.optimize import linear_sum_assignment


class MetricManager:

    """
    Main class that keeps track and computes metrics through mini-batches through an epoch.
    """

    def __init__(self, metric_names: list[str]):
        self.metric_names = metric_names
        self.epoch_metrics = None
        self.non_textual_tokens = get_sem_tokens_read2016() + get_sem_tokens_rimes2009()

        # association between metric names and submetrics requirements
        self.linked_metrics = {
            "cer": ["edit_chars", "nb_chars"],
            "wer": ["edit_words", "nb_words"],
            "cer_force_len": ["edit_chars_force_len", ],
            "cer_by_nb_lines": ["nb_lines", ],
            "cer_by_nb_cols": ["nb_cols", ],
            "worse_cer": [],
            "flex_cer": ["flex_edit_chars", "flex_nb_chars"],
            "flex_wer": ["flex_edit_words", "flex_nb_words"],
        }
        self.init_metrics()

    def init_metrics(self) -> None:
        """
        Initialization of the metrics specified in metrics_name
        """

        self.epoch_metrics = {
            "nb_samples": list(),
            "names": list(),
            "ids": list(),
        }

        for metric_name in self.metric_names:
            if metric_name in self.linked_metrics:
                for linked_metric_name in self.linked_metrics[metric_name]:
                    if linked_metric_name not in self.epoch_metrics.keys():
                        self.epoch_metrics[linked_metric_name] = list()
            else:
                self.epoch_metrics[metric_name] = list()

    def update_metrics(self, batch_metrics: dict[str, list]) -> None:
        """
        Aggregate batch metrics to current epoch metrics.
        :param batch_metrics: dict associating metric names (str) to sample values (as a list).
        """
        for key in batch_metrics.keys():
            if key in self.epoch_metrics:
                self.epoch_metrics[key] += batch_metrics[key]

    def get_display_values(self, output: bool=False) -> dict:
        """
        Format metrics values for shell display purposes.
        :param output: whether to output the result in a file or not (True at evaluation time).
        :return: a dict with metric names as keys, and associated value as values.
        """
        metric_names = self.metric_names.copy()
        if output:
            metric_names.extend(["nb_samples"])
        display_values = dict()
        for metric_name in metric_names:
            value = None
            if output:
                if metric_name in ["nb_samples", "weights"]:
                    value = np.sum(self.epoch_metrics[metric_name])
                elif metric_name in ["time", ]:
                    total_time = np.sum(self.epoch_metrics[metric_name])
                    sample_time = total_time / np.sum(self.epoch_metrics["nb_samples"])
                    display_values["sample_time"] = round(sample_time, 4)
                    value = total_time
            if metric_name == "cer":
                value = np.sum(self.epoch_metrics["edit_chars"]) / np.sum(self.epoch_metrics["nb_chars"])
                if output:
                    display_values["nb_chars"] = np.sum(self.epoch_metrics["nb_chars"])
            elif metric_name == "wer":
                value = np.sum(self.epoch_metrics["edit_words"]) / np.sum(self.epoch_metrics["nb_words"])
                if output:
                    display_values["nb_words"] = np.sum(self.epoch_metrics["nb_words"])
            elif metric_name == "flex_cer":
                value = np.sum(self.epoch_metrics["flex_edit_chars"]) / np.sum(self.epoch_metrics["flex_nb_chars"])
            elif metric_name == "flex_wer":
                value = np.sum(self.epoch_metrics["flex_edit_words"]) / np.sum(self.epoch_metrics["flex_nb_words"])
            elif metric_name in ["diff_len", ]:
                value = np.mean(np.abs(self.epoch_metrics[metric_name]))
            elif metric_name in ["", ]:
                value = np.sum(self.epoch_metrics[metric_name]) / self.epoch_metrics["nb_samples"]
            elif metric_name in ["loss", "loss_ctc", "loss_ce", "loss_main", "loss_weights", "syn_max_lines", "syn_ratio"]:
                value = np.average(self.epoch_metrics[metric_name], weights=np.array(self.epoch_metrics["nb_samples"]))
            elif value is None:
                continue

            display_values[metric_name] = round(value, 4)
        return display_values

    def compute_metrics(self, values: dict, metric_names: list[str]) -> dict:
        """
        Compute metrics over output values from a mini-batch.
        """
        metrics = {
            "nb_samples": [values["nb_samples"], ],
        }
        for v in ["weights", "time"]:
            if v in values:
                metrics[v] = [values[v]]
        for v in ["nb_lines", "nb_cols"]:
            if v in values:
                metrics[v] = values[v]
        for metric_name in metric_names:
            if metric_name == "cer":
                metrics["edit_chars"] = [edit_cer_from_string(u, v, self.non_textual_tokens) for u, v in zip(values["str_y"], values["str_x"])]
                metrics["nb_chars"] = [nb_chars_cer_from_string(gt, self.non_textual_tokens) for gt in values["str_y"]]
            elif metric_name == "wer":
                split_gt = [format_string_for_wer(gt, self.non_textual_tokens) for gt in values["str_y"]]
                split_pred = [format_string_for_wer(pred, self.non_textual_tokens) for pred in values["str_x"]]
                metrics["edit_words"] = [edit_wer_from_formatted_split_text(gt, pred) for (gt, pred) in zip(split_gt, split_pred)]
                metrics["nb_words"] = [len(gt) for gt in split_gt]
            elif metric_name == "flex_cer":
                metrics["flex_edit_chars"] = [flex_edit_cer_from_string(u, v, self.non_textual_tokens) for u, v in zip(values["str_y"], values["str_x"])]
                metrics["flex_nb_chars"] = [flex_nb_chars_cer_from_string(gt, self.non_textual_tokens) for gt in values["str_y"]]
            elif metric_name == "flex_wer":
                metrics["flex_edit_words"] = [flex_edit_wer_from_string(u, v, self.non_textual_tokens) for u, v in zip(values["str_y"], values["str_x"])]
                metrics["flex_nb_words"] = [len(gt) for gt in split_gt]
            elif metric_name == "pred":
                metrics[metric_name] = values["str_x"]
            elif metric_name in ["loss_ctc", "loss_ce", "loss", "loss_main", "loss_weights", "syn_max_lines", "syn_ratio"]:
                metrics[metric_name] = [values[metric_name], ]
            elif metric_name in ["diff_len", ]:
                metrics[metric_name] = values[metric_name]
        return metrics

    def get(self, name: str) -> list:
        """
        Getter for metric values as a list (one value per sample)
        """
        return self.epoch_metrics[name]


def keep_only_tokens(str: str, tokens: str) -> str:
    """
    Remove all but layout tokens from string
    """
    return re.sub('([^' + tokens + '])', '', str)


def keep_all_but_tokens(str: str, tokens: str) -> str:
    """
    Remove all layout tokens from string
    """
    return re.sub('([' + tokens + '])', '', str)


def edit_cer_from_string(gt: str, pred: str, non_textual_tokens: str=None) -> int:
    """
    Format and compute edit distance between two strings at character level
    """
    gt = format_string_for_cer(gt, non_textual_tokens)
    pred = format_string_for_cer(pred, non_textual_tokens)
    return editdistance.eval(gt, pred)

def flex_edit_cer_from_string(gt: str, pred: str, non_textual_tokens: str=None) -> int:
    """
    Format and compute edit distance between two strings at character level (considering the best line ordering)
    """
    gt = format_string_for_cer(gt, non_textual_tokens)
    pred = format_string_for_cer(pred, non_textual_tokens)
    gt_lines = gt.split("\n")
    pred_lines = pred.split("\n")
    return hungarian_matching(pred_lines, gt_lines)

def hungarian_matching(pred_list: list[str], gt_list: list[str], level: str="char"):
    """
    Compute edit distance between two list of predictions by finding the best matching using hungarian algorithm.
    """
    while len(pred_list) > len(gt_list):
        gt_list.append("" if level == "char" else [])
    while len(pred_list) < len(gt_list):
        pred_list.append("" if level == "char" else [])
    scores = []
    for pred in pred_list:
        scores.append([editdistance.eval(gt, pred) for gt in gt_list])
    scores = np.array(scores)
    row_ind, col_ind = linear_sum_assignment(scores)
    return scores[row_ind, col_ind].sum()
    


def nb_chars_cer_from_string(gt: str, non_textual_tokens: str=None) -> int:
    """
    Compute length after formatting of ground truth string
    """
    return len(format_string_for_cer(gt, non_textual_tokens))


def flex_nb_chars_cer_from_string(gt: str, non_textual_tokens: str=None) -> int:
    """
    Compute length after formatting of ground truth string for flex cer computation (removing line breaks)
    """
    formatted_str = format_string_for_cer(gt, non_textual_tokens).replace("\n", "")
    return len(formatted_str)


def edit_wer_from_string(gt: str, pred: str, non_textual_tokens: str=None) -> int:
    """
    Format and compute edit distance between two strings at word level
    """
    split_gt = format_string_for_wer(gt, non_textual_tokens)
    split_pred = format_string_for_wer(pred, non_textual_tokens)
    return edit_wer_from_formatted_split_text(split_gt, split_pred)

def flex_edit_wer_from_string(gt: str, pred: str, non_textual_tokens: str=None) -> int:
    """
    Format and compute edit distance between two strings at word level
    """
    gt_lines = gt.split("\n")
    pred_lines = pred.split("\n")
    gt_lines = [format_string_for_wer(gt_line, non_textual_tokens) for gt_line in gt_lines]
    pred_lines = [format_string_for_wer(pred_line, non_textual_tokens) for pred_line in pred_lines]
    return hungarian_matching(pred_lines, gt_lines, level="word")


def format_string_for_wer(str: str, non_textual_tokens: str) -> str:
    """
    Format string for WER computation: remove layout tokens, treat punctuation as word, replace line break by space
    """
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)  # punctuation processed as word
    if non_textual_tokens is not None:
        str = keep_all_but_tokens(str, non_textual_tokens)  # remove layout tokens from metric
    str = re.sub('([ \n])+', " ", str).strip()  # keep only one space character
    return str.split(" ")


def format_string_for_cer(str: str, non_textual_tokens: str) -> str:
    """
    Format string for CER computation: remove layout tokens and extra spaces
    """
    if non_textual_tokens is not None:
        str = keep_all_but_tokens(str, non_textual_tokens)  # remove layout tokens from metric
    str = re.sub('([\n])+', "\n", str)  # remove consecutive line breaks
    str = re.sub('([ ])+', " ", str).strip()  # remove consecutive spaces
    return str


def edit_wer_from_formatted_split_text(gt: list[str], pred: list[str]) -> int:
    """
    Compute edit distance at word level from formatted string as list of words
    """
    return editdistance.eval(gt, pred)