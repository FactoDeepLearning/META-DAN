from datasets import load_dataset
import os
from random import shuffle
from tqdm import tqdm
import re


class WikiManager:

    # Chinese, Japanese and Korean characters
    cjk_char_regexp = re.compile(r'[\u4e00-\u9fff]+')

    def __init__(self, language, max_chars_per_line):
        self.max_chars_per_line = max_chars_per_line
        self.data_path = os.path.join("Datasets", "formatted", "wiki", f"wiki_{language}_1M.txt")
        self.data = None
        self.language = language
        self.load_data()
        self.split_samples_by_max_num_chars()

    def load_data(self) -> None:
        """
        Load or download/format language-specific wikipedia dataset
        """
        if os.path.exists(self.data_path):
            with open(self.data_path, "r") as f:
                self.data = f.read().split("\n")
        else:
            self.load_raw_data()
            self.generate_sample_file(self.data_path)

    def filter_data_by_charset(self, charset: list[str]) -> None:
        """
        Only keep wikipedia lines that match with given charset
        """
        self.data = [text for text in self.data if WikiManager.is_valid_text(text, charset)]
        print(f"Num sentences kept from {self.data_path}: {len(self.data)}")

    @staticmethod
    def is_valid_text(text: str, charset: list[str]) -> bool:
        """
        Return True if all characters of text belongs to the given charset.
        """
        return set(text).issubset(set(charset))

    def load_raw_data(self):
        """
        Load wikipedia dataset through wikimedia repo.
        """
        self.data = load_dataset("wikimedia/wikipedia", f"20231101.{self.language}", trust_remote_code=True, cache_dir=".cache", num_proc=4)

    def generate_sample_file(self, out_path: str, num_lines: int=1000000) -> None:
        """
        Generate a clean sample file at out_path from wikipedia source with given number of text lines
        """
        lines = list()
        i = 0
        with tqdm(total=num_lines) as progress_bar:
            progress_bar.set_description("Building wiki text file...")
            while len(lines) < num_lines:
                new_lines = self.data["train"][i]["text"].replace(". ", ".\n").split("\n")
                new_lines = [l.strip() for l in new_lines if len(l.strip() .split(" ")) > 3]
                if self.language == "zh":
                    new_lines = [l for l in new_lines if WikiManager.cjk_char_regexp.match(l)]
                shuffle(new_lines)
                lines.extend(new_lines[:100])  # max 100 lines from the same article
                i += 1
                progress_bar.update(len(new_lines))
        lines = lines[:num_lines]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        self.data = lines

    def split_samples_by_max_num_chars(self, word_sep: str=" ") -> None:
        """
        Split the text lines in order for them to have at most self.max_chars_per_line characters.
        Splitting is performed at word level.
        """
        new_samples = list()
        for sample in self.data:
            if len(sample) > self.max_chars_per_line:
                words = sample.split(" ") if word_sep else sample
                while True:
                    new_sample = []
                    while len(words) > 0 and len(word_sep.join(new_sample)) < self.max_chars_per_line:
                        new_sample.append(words.pop(0))
                    if len(new_sample) == 0:
                        break
                    new_samples.append(word_sep.join(new_sample))
        self.data = new_samples

