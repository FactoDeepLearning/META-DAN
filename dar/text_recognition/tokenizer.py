
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers import pre_tokenizers
from tokenizers import Tokenizer as HF_Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder
import os
import json
import re

from dar.dataset_format.read2016 import get_sem_tokens as get_sem_tokens_read2016
from dar.dataset_format.rimes2009 import get_sem_tokens as get_sem_tokens_rimes2009
from dar.text_recognition.synthetic.wikipedia_manager import WikiManager


class Tokenizer:

    def __init__(self, config):
        self.config = config
        self.invisible_chars = set(" \n" + get_sem_tokens_rimes2009() + get_sem_tokens_read2016())

    def text_to_token(self, text):
        pass

    def token_to_text(self, token):
        pass

    @staticmethod
    def preprocess_ctc(indices):
        res = []
        for i in indices:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res


class CharacterTokenizer(Tokenizer):

    def __init__(self, config: dict):
        super().__init__(config)
        self.charset = self.config["charset"]
        self.newline_char = "\n"

        # To check fonts (keep only writable chars)
        self.char_only_set = self.charset.copy()
        for token in self.invisible_chars:
            if token in self.char_only_set:
                self.char_only_set.remove(token)

        self.tokens_list = self.charset.copy()
        self.num_text_tokens = len(self.charset)
        self.tokens_list.extend(config["additional_tokens"])
        self.tokens_to_indices = {c: i for i, c in enumerate(self.tokens_list)}
        self.indices_to_tokens = {v: k for k, v in self.tokens_to_indices.items()}
        self.num_tokens = len(self.tokens_list) - 1  # padding token should not be predicted

    @property
    def tokens(self):
        return self.tokens_to_indices

    @property
    def text_tokens(self):
        return self.charset

    @property
    def text_indices(self):
        return [self.tokens_to_indices[token] for token in self.text_tokens]

    def text_to_token(self, tokens):
        return [self.tokens_to_indices[token] for token in tokens]

    def token_to_text(self, indices):
        return "".join([self.indices_to_tokens[int(index)] for index in indices if index in self.text_indices])


class BPETokenizer(Tokenizer):

    def __init__(self, config):
        super().__init__(config)
        self.num_tokens = config["tok_num_tokens"]
        self.token_mode = "bpe"
        self.storing_dir = ".cache"
        self.newline_char = "Ⓩ"
        self.bpe_model = None

        self.model_name = config["model_name"]
        if self.model_name is None:
            if config["bpe_source"] == "dataset":
                self.train_file_path = self.generate_dataset_training_txt_file()
            else:
                language = config["bpe_source"].split("-")[-1]
                self.train_file_path = self.generate_wiki_training_txt_file(language)
            self.model_name = os.path.join(self.storing_dir, f"tokenizers_bpe_{config['dataset_name']}_{config['bpe_source']}_{self.num_tokens}")
        else:
            self.model_name = os.path.join(self.storing_dir, self.model_name)

        if not os.path.exists(f"{self.model_name}"):
            self.train_bpe()
        else:
            self.bpe_model = HF_Tokenizer.from_file(self.model_name)

        # To check fonts (keep only writable chars)
        self.char_only_set = self.config["charset"].copy()
        for token in self.invisible_chars:
            if token in self.char_only_set:
                self.char_only_set.remove(token)

        self.charset = list(self.bpe_model.get_vocab().items())
        self.charset.sort(key=lambda x: x[1])
        self.charset = [v[0] for v in self.charset]

        self.tokens_list = self.charset.copy()
        self.num_text_tokens = len(self.charset)
        self.tokens_list.extend(config["additional_tokens"])
        self.tokens_to_indices = {c: i for i, c in enumerate(self.tokens_list)} | {"\n": self.tokens_list.index(f"{self.newline_char}</w>")}
        self.indices_to_tokens = {v: k for k, v in self.tokens_to_indices.items()}
        self.num_tokens = len(self.tokens_list) - 1  # padding token should not be predicted

    @property
    def tokens(self):
        return self.tokens_to_indices

    @property
    def text_tokens(self):
        return self.charset

    @property
    def text_indices(self):
        return [self.tokens_to_indices[token] for token in self.text_tokens]

    def text_to_token(self, tokens):
        if isinstance(tokens, list):
            tokens = "".join(tokens)
        return self.bpe_model.encode(tokens.replace("\n", f" {self.newline_char} ")).ids

    def token_to_text(self, indices):
        indices = [int(index) for index in indices if index in self.text_indices]
        text = self.bpe_model.decode(indices)
        return re.sub(f"(\s*{self.newline_char}\s*)", "\n", text) 

    def generate_dataset_training_txt_file(self):
        data = list()
        for path in self.config["dataset_paths"]:
            with open(os.path.join(path, "labels.json")) as f:
                labels = json.load(f)
            for file_path, sample in labels["train"].items():
                data.append("".join(sample["text"]))
        data = "\n".join(data)
        os.makedirs(self.storing_dir, exist_ok=True)
        train_file_path = os.path.join(self.storing_dir, "train_text.txt")
        with open(train_file_path, "w") as f:
            f.write(data.replace("\n", f" {self.newline_char} "))
        return train_file_path
    
    def generate_wiki_training_txt_file(self, language):
        wiki = WikiManager(language, max_chars_per_line=100)
        wiki.filter_data_by_charset("".join(self.config["charset"]))
        train_file_path = os.path.join(self.storing_dir, "train_text_wiki.txt")
        with open(train_file_path, "w") as f:
            f.write(f" {self.newline_char} ".join(wiki.data))
        return train_file_path

    def train_bpe(self):
        print("Training BPE... (this can take a moment)")
        self.bpe_model = HF_Tokenizer(BPE())
        self.bpe_model.pre_tokenizer = pre_tokenizers.Sequence([WhitespaceSplit(),
                                                          # Digits(individual_digits=True)
                                                          ])
        trainer = BpeTrainer(vocab_size=self.num_tokens,
                             show_progress=True,
                             initial_alphabet=self.config["charset"],
                             # continuing_subword_prefix="##",
                             end_of_word_suffix="</w>",
                             max_token_length=5,
                             )
        self.bpe_model.train([self.train_file_path, ], trainer)
        self.bpe_model.decoder = BPEDecoder(suffix="</w>")
        self.bpe_model.save(self.model_name)


    @staticmethod
    def pre_tokenizer(text):
        return text.replace("\n", " \n ").split(" ")
