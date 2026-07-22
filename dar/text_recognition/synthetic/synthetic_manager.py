from dar.text_recognition.synthetic.synthetic_doc import SyntheticPage, SyntheticBlock
from dar.text_recognition.synthetic.wikipedia_manager import WikiManager
from dar.text_recognition.synthetic.synthetic_read2016 import READ2016Synthetic
from dar.text_recognition.synthetic.synthetic_rimes import RIMESSynthetic
from dar.text_recognition.synthetic.synthetic_maurdor import MAURDORSynthetic
from dar.text_recognition.synthetic.font_manager import FontManager
from dar.text_recognition.synthetic.synthetic_line import generate_random_text_line_label, generate_typed_text_line_image_default_config
from dar.generic.scheduler import linear_scheduler
from dar.generic.utils import randint, rand, randint_normal
from scipy.ndimage import rotate
import copy
import numpy as np


class SyntheticManager:
    """
    Main class to handle synthetic data generation and curriculum strategy
    """

    def __init__(self, dataset):
        self.params = dataset.params
        self.dataset = dataset
        self.font_manager = FontManager({"syn_num_fonts": self.params["config"]["syn_num_fonts"]})
        self.wiki_managers = {}
        self.doc_generators = {}
        self.sample_ids_per_dataset = {}
        self.init_wiki_managers()
        self.init_doc_generators()
        self.init_sample_ids_per_dataset()

    def apply_synthetic(self, sample: dict, training_info: dict) -> dict:
        """
        Apply the synthetic strategy: randomly generate a synthetic sample given the configuration.
        """
        # Update training info
        self.dataset.training_info = training_info

        if rand() > self.get_syn_ratio(sample):
            return sample

        return self.generate_synthetic_sample(sample)
    
    def init_sample_ids_per_dataset(self) -> dict:
        """
        Compute the list of samples ids that correpond to each dataset
        """
        sample_ids_per_dataset = {}
        for i, s in enumerate(self.dataset.samples):
            dataset = s["dataset"]
            if dataset not in sample_ids_per_dataset:
                sample_ids_per_dataset[dataset] = [i, ]
            else:
                sample_ids_per_dataset[dataset].append(i)
        self.sample_ids_per_dataset = sample_ids_per_dataset

    def init_doc_generators(self) -> None:
        """
        Initialize the specific synthetic document generators to be used in training (it depends on the training datasets).
        """
        if "custom_config" in self.params["config"]["synthetic_data"]:
            generator_configs_per_dataset = self.params["config"]["synthetic_data"]["custom_config"]
            for dataset_name, generator_configs in generator_configs_per_dataset.items():
                for generator_config in generator_configs:
                    generator_name = generator_config["name"]
                    if generator_name == "read2016":
                        self.doc_generators[generator_name] = READ2016Synthetic(self.font_manager,
                                                                    generator_config["label_path"],
                                                                    add_sem_tokens=generator_config["add_sem_tokens"])
                    elif generator_name == "rimes":
                        enable_overlap = generator_config["enable_overlap"] if "enable_overlap" in generator_config else False
                        self.doc_generators[generator_name] = RIMESSynthetic(self.font_manager,
                                                                    generator_config["label_path"],
                                                                    add_sem_tokens=generator_config["add_sem_tokens"],
                                                                    enable_overlap=enable_overlap)
                    elif generator_name == "maurdor":
                        self.doc_generators[generator_name] = MAURDORSynthetic(self.font_manager,
                                                                    generator_config["label_path"])
    
    def get_syn_ratio(self, sample: dict=None) -> float:
        """
        Compute the probability of generating and using a synthetic document for training with respect to the current state of the training.
        """
        synthetic_params = self.params["config"]["synthetic_data"]
        scheduler = synthetic_params["scheduler"]
        nb_samples = self.dataset.training_info["num_samples"]
        syn_ratio_info = copy.deepcopy(scheduler)
        syn_ratio_info.update(synthetic_params)
        if "curriculum" in synthetic_params:
            syn_ratio_info.update(synthetic_params["curriculum"])
        
        if sample is not None:
            dataset_specific_config = [x for x in self.params["train"] if x["dataset_name"] == sample["dataset"]][0]
            for key in ["init_proba", "end_proba", "num_samples_curr", "start_curr_at_sample", "start_scheduler_after_curriculum", "num_samples_proba"]:
                if key in dataset_specific_config:
                    syn_ratio_info.update({key: dataset_specific_config[key]})
        
        if syn_ratio_info["init_proba"] == syn_ratio_info["end_proba"] == 1:
            proba = 1
        elif "start_scheduler_after_curriculum" in syn_ratio_info and syn_ratio_info["start_scheduler_after_curriculum"]:
            nb_samples_for_scheduler = max(0, nb_samples-syn_ratio_info["num_samples_curr"]-syn_ratio_info["start_curr_at_sample"])
            if nb_samples_for_scheduler <= 0:
                current_step = 0
            else:
                current_step = min(syn_ratio_info["num_samples_proba"], nb_samples_for_scheduler)
            proba = linear_scheduler(init_value=syn_ratio_info["init_proba"], end_value=syn_ratio_info["end_proba"], step=current_step, max_step=syn_ratio_info["num_samples_proba"])
        else:
            current_step = min(syn_ratio_info["num_samples_proba"], nb_samples)
            proba = linear_scheduler(init_value=syn_ratio_info["init_proba"], end_value=syn_ratio_info["end_proba"], step=current_step, max_step=syn_ratio_info["num_samples_proba"])
        return proba

    def generate_synthetic_sample(self, sample: dict) -> dict:
        """
        Generate a synthetic sample based on a real one (given as parameter). 
        Depending on the configuraiton, it will generate a line, a text block or a page.
        :param sample: an original sample from training dataset.
        """
        mode = self.params["config"]["synthetic_data"]["mode"]
        text_source = self.params["config"]["synthetic_data"]["text_source"]
        if text_source == "dataset":
            text_source = f"dataset-{sample['dataset']}"
        sample["is_synthetic"] = True
        assert mode in ["line", "text-block", "page"]

        if mode == "line":
            return self.generate_synthetic_line(sample, text_source)

        if mode == "text-block":
            return self.generate_synthetic_text_block(sample, text_source)

        if mode == "page":
            return self.generate_synthetic_page(sample)

    def generate_synthetic_line(self, sample: dict, text_source: str, word_sep: str=" ") -> dict:
        """
        Generate a synthetic line sample based on a real one (given as parameter)
        :param sample: an original sample from training dataset.
        :param text_source: text data from which to randomly pick to generate the synthetic sample ("dataset" or "wiki-XX")
        :param word_sep: character used a word separator ("" for chinese).
        """
        synthetic_params = self.params["config"]["synthetic_data"]
        text = self.generate_text_line_label(text_source, word_sep)
        if synthetic_params["syn_double_line"]:
            text = f"{text} {self.generate_text_line_label(text_source, word_sep)}"

        # curriculum at char level for single-line syn generation
        if "scheduler_char" in synthetic_params:
            samples_per_char = synthetic_params["scheduler_char"]["num_samples"]/(synthetic_params["char_per_line"]["max"] - synthetic_params["char_per_line"]["min"])
            max_chars = self.dataset.training_info["num_samples"] / samples_per_char
            max_chars = int(min(max(max_chars, 1), synthetic_params["char_per_line"]["max"]))
            text = text[:max_chars]

        sample = self.dataset.update_sample_labels(sample, text)
        text = "".join(text)
        sample["img"] = generate_typed_text_line_image_default_config(text=text, font_path=self.font_manager.get_random_font(text), max_width=synthetic_params["max_width"])
        if "rotate_proba" in synthetic_params and rand() < synthetic_params["rotate_proba"]:
            rotation_degree = randint_normal(0, 15, -45, 45)
            sample["img"] = rotate(sample["img"], angle=rotation_degree, reshape=True, cval=self.padding_value)
        return sample

    def generate_synthetic_text_block(self, sample: dict, text_source: str) -> str:
        """
        Generate a synthetic text block (single column of text) based on a real one (given as parameter)
        :param sample: an original sample from training dataset.
        :param text_source: text data from which to randomly pick to generate the synthetic sample ("dataset" or "wiki-XX")
        """
        max_num_lines = self.get_syn_max_lines()
        num_lines = randint(1, max_num_lines+1)
        if num_lines == 1:
            return self.generate_synthetic_line(sample, text_source)
        text_lines = ["".join(self.generate_text_line_label(text_source)) for _ in range(num_lines)]
        width = randint(1000, 1500)
        height = min(randint(40*num_lines, 70*num_lines), 1800)
        text_block = SyntheticBlock(height=height, width=width, font_manager=self.font_manager)
        text = " ".join(text_lines)
        text_block.render(text, sample["dataset"])
        label = list("\n".join(text_block.text_lines))
        sample = self.dataset.update_sample_labels(sample, label)
        sample["img"] = text_block.image
        return sample

    def generate_synthetic_page(self, sample: dict) -> dict:
        """
        Generate a synthetic page based on a real one (given as parameter)
        :param sample: an original sample from training dataset.
        """
        gen_configs = self.params["config"]["synthetic_data"]["custom_config"][sample["dataset"]]
        probs = [gen_config["proba"] for gen_config in gen_configs]
        gen_config = np.random.choice(gen_configs, p=probs)
        text_source = gen_config["text_source"] if "text_source" in gen_config else "dataset"
        if text_source == "dataset":
            text_source = f"dataset-{sample['dataset']}"
        max_num_lines = self.get_syn_max_lines(absolute_max_lines=gen_config["max_num_lines"])
        num_lines = randint(1, max_num_lines+1)


        word_sep = gen_config["word_sep"]  if "word_sep" in gen_config else " "
        if gen_config["name"] == "generic":
            if num_lines == 1:
                return self.generate_synthetic_line(sample, text_source, word_sep)
            ratio_num_columns = gen_config["ratio_num_columns"] if "ratio_num_columns" in gen_config else [0.8, 0.2]
            image, label = self.generate_generic_page(sample, num_lines, text_source, 
                                                      ratio_num_columns=ratio_num_columns, 
                                                      word_sep=word_sep)
        else:
            image, label = self.generate_custom_page(sample, gen_config["name"], num_lines, max_num_lines, 
                                                     source=gen_config["text_source"], 
                                                     word_sep=word_sep)

        sample = self.dataset.update_sample_labels(sample, label)
        sample["img"] = image
        return sample
    
    def generate_custom_page(self, sample: dict, 
                             generator_name: str, 
                             num_lines: int, 
                             max_num_lines: int, 
                             source: str, 
                             word_sep: str=" ") -> str:
        """
        For some datasets, custom synthetic generators are available (RIMES, MAURDOR, READ 2016) refered as doc_generators.
        :param sample: an original sample from training dataset.
        :param source: text data from which to randomly pick to generate the synthetic sample ("dataset" or "wiki-XX").
        :param word_sep: character used a word separator ("" for chinese).
        :param num_lines: number of lines to be included in the syntehtic page to be generated.
        :param max_num_lines: maximum number of lines that could have been selected in the current state of the training (for curriculum strategies).
        """
        page_generator = self.doc_generators[generator_name]
        text_lines = None if "dataset" in source else self.generate_text_line_labels(source, 100, word_sep)
        return page_generator.render(sample,
                                     [num_lines, ] if "read" in generator_name else num_lines,
                                     crop=max_num_lines != self.params["config"]["synthetic_data"]["max_num_lines"],
                                     text_lines=text_lines)
    
    def generate_generic_page(self, sample: dict, 
                              num_lines: int, 
                              text_source: str, 
                              ratio_num_columns: tuple=[1.0, ], 
                              word_sep: str=" "):
        """
        For many datasets, we used a generic layout to generate synthetic documents.
        :param sample: an original sample from training dataset.
        :param text_source: text data from which to randomly pick to generate the synthetic sample ("dataset" or "wiki-XX").
        :param word_sep: character used a word separator ("" for chinese).
        :param num_lines: number of lines to be included in the syntehtic page to be generated.
        :param ratio_num_columns: [1.0] means 100% of single column documents, [0.3, 0.7] means 30% of single-column documents and 70% of double-column documents
        """
        text_lines = self.generate_text_line_labels(text_source, num_lines, word_sep)
        width = randint(1000, 1500)
        height = min(randint(40*num_lines, 70*num_lines), 1800)
        page = SyntheticPage(height=height, width=width, font_manager=self.font_manager, ratio_num_cols=ratio_num_columns)
        text = word_sep.join(text_lines)
        page.render(text, sample["dataset"], word_sep=word_sep)
        label = list("\n".join(page.text_lines))
        image = page.image
        return image, label


    def generate_text_line_label(self, text_source: str, word_sep: str=" ") -> str:
        """
        Generate a single text line label from the given text source.
        :param text_source: text data from which to randomly pick to generate the synthetic sample ("dataset" or "wiki-XX").
        :param word_sep: character used a word separator ("" for chinese).
        """
        return self.generate_text_line_labels(text_source, 1, word_sep)[0]

    def generate_text_line_labels(self, text_source: str, num_lines: int=1, word_sep: str=" ") -> list[str]:
        """
        Generate some text line labels from the given text source.
        :param text_source: text data from which to randomly pick to generate the synthetic sample ("dataset" or "wiki-XX").
        :param word_sep: character used a word separator ("" for chinese).
        :param num_lines: number of labels to generate.
        """
        synthetic_params = self.params["config"]["synthetic_data"]
        if text_source == "random":
            return [generate_random_text_line_label(self.tokenizer.char_only_set,
                                            synthetic_params["char_per_line"]["min"],
                                            synthetic_params["char_per_line"]["max"]) for _ in range(num_lines)]
        return self.generate_successive_text_line_labels(text_source, num_lines, word_sep)

    def get_random_num_chars_line_syn(self):
        """
        Randomly pick the size of the label to generate following configuration.
        """
        config = self.params["config"]["synthetic_data"]["char_per_line"]
        return randint_normal(mean=config["mean"], 
                              std=config["std"], 
                              min=config["min"], 
                              max=config["max"])
    
    def get_random_sample(self, dataset: str|None=None) -> dict:
        """
        Randomly select one sample from the dataset.
        :param dataset: for multi-dataset training, one can specify the dataset from which to select the sample.
        """
        if dataset is None:
            sample_index = randint(0, len(self.dataset.samples))
        else:
            dataset = dataset.replace("dataset-", "")
            num_samples = len(self.sample_ids_per_dataset[dataset])
            sample_id_index = randint(0, num_samples)
            sample_index = self.sample_ids_per_dataset[dataset][sample_id_index]
        return self.dataset.samples[sample_index]

    def generate_successive_text_line_labels(self, text_source: str, num_lines: int=1, word_sep: str=" "):
        """
        Generate some text line labels whose content follow one another in the given source.
        :param text_source: text data from which to randomly pick to generate the synthetic sample ("dataset" or "wiki-XX").
        :param word_sep: character used a word separator ("" for chinese).
        :param num_lines: number of labels to generate.
        """
        synthetic_params = self.params["config"]["synthetic_data"]
        if "wiki" in text_source and text_source not in self.wiki_managers:
            self.init_wiki_manager(text_source)
        lines = []
        current_text = None
        while len(lines) < num_lines:
            if current_text is None:
                if "dataset" in text_source:
                    sample = self.get_random_sample(text_source)
                    sample = "".join(sample["gt"]["text"])
                    sample_lines = [l for l in sample.split("\n") if len(l) >= synthetic_params["char_per_line"]["min"]]
                    if len(sample_lines) <= 0:
                        continue
                    line_index = randint(0, len(sample_lines))
                    current_text = word_sep.join(sample_lines[line_index:])
                else: 
                    line_index = randint(0, len(self.wiki_managers[text_source].data))
                    current_text = word_sep.join(self.wiki_managers[text_source].data[line_index:line_index+10])
            num_chars = self.get_random_num_chars_line_syn()
            # try avoid splitting words
            if word_sep and len(current_text) > num_chars + 1 and current_text[num_chars+1] != word_sep:
                for _ in range(10): 
                    if len(current_text) > num_chars + 1 and current_text[num_chars+1]:
                        num_chars += 1
                    if len(current_text) == num_chars + 1 or current_text[num_chars+1] == word_sep:
                        break
            new_line = current_text[:num_chars+1].strip()
            if len(new_line) > 0:
                lines.append(new_line)
            if len(current_text) <= num_chars:
                current_text = None
            else:
                current_text = current_text[num_chars+1:]

            # Add duplicated lines in syn
            if "add_loop_syn" in synthetic_params and synthetic_params["add_loop_syn"] and len(lines) > 1 and rand() < 0.05:
                duplicated_line_index = randint(0, len(lines))
                replaced_index = randint(0, len(lines))
                while replaced_index == duplicated_line_index:
                    replaced_index = randint(0, len(lines))
                lines[replaced_index] = lines[duplicated_line_index]
        return lines
    
    
    def init_wiki_manager(self, text_source: str) -> None:
        """
        Initialize a given wiki manager 
        :param text_source: "wiki-XX" with XX the language (fr for French, en for English)
        """
        synthetic_params = self.params["config"]["synthetic_data"]
        language = text_source.replace("wiki-", "")
        max_chars = synthetic_params["char_per_line"]["max"]
        self.wiki_managers[text_source] = WikiManager(language, max_chars)
        self.wiki_managers[text_source].filter_data_by_charset(self.get_charset())

    def init_wiki_managers(self) -> None:
        """
        Initialize all necessary wiki managers
        """
        if "synthetic_data" in self.params["config"] and "custom_config" in self.params["config"]["synthetic_data"]:
            for dataset_name, configs in self.params["config"]["synthetic_data"]["custom_config"].items():
                for config in configs:
                    if "wiki" in config["text_source"] and config["text_source"] not in self.wiki_managers:
                        self.init_wiki_manager(config["text_source"])

    def get_syn_max_lines(self, absolute_max_lines: int=None, absolute_min_lines: int=None) -> int:
        """
        Compute the maximum number of lines per page that can be used for synthetic generation at the current state of the training (taking into account curriculum learning)
        :param absolute_max_lines: the upper bound limit.
        :param absolute_min_lines: the lower bound limit.
        """
        synthetic_params = self.params["config"]["synthetic_data"]
        if synthetic_params["mode"] == "line":
            return 1
        absolute_max_lines = synthetic_params["max_num_lines"] if absolute_max_lines is None else absolute_max_lines
        absolute_min_lines = synthetic_params["curriculum"]["min_num_lines"] if absolute_min_lines is None else absolute_min_lines
        if "curriculum" in synthetic_params and synthetic_params["curriculum"] and synthetic_params["curriculum"]["num_samples_curr"] > 0 and absolute_min_lines < absolute_max_lines:
            curr_samples = synthetic_params["curriculum"]["num_samples_curr"]/(absolute_max_lines-absolute_min_lines)
            nb_samples_for_curr = max(0, self.dataset.training_info["num_samples"] - synthetic_params["start_curr_at_sample"])
            num_steps = int(nb_samples_for_curr/curr_samples)
            max_nb_lines = min(absolute_min_lines + num_steps, absolute_max_lines)
            return max_nb_lines
        return absolute_max_lines