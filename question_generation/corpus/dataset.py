""" Script to load and process the corpus that will be used in the QA generation process. """
from datasets import Dataset, load_dataset
from pathlib import Path
from typing import Generator

OUTPUT_DATASETS_DIR = Path(__file__).resolve(
).parents[2] / 'output_datasets'  # two levels up


class DatasetLoader:
    """ Class to load Hugging Face datasets. """

    def __init__(self, dataset_name: str, split: str = 'train'):
        self.dataset_name = dataset_name
        self.split = split
        self._dataset = load_dataset(self.dataset_name)

    @property
    def dataset(self) -> Dataset:
        return self._dataset
    
    @property
    def dataset_length(self) -> int:
        return len(self._dataset[self.split])

    def get_data(self) -> Generator[dict, None, None]:
        """ Returns a generator with the data from the dataset. """
        for data in self._dataset[self.split]:
            yield data


class DatasetBuilder:
    """ Class to build a Hugging Face dataset. It is used to create a dataset based on chat QA generated data and have the capability to resume the process if it is interrupted. """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.output_dir = OUTPUT_DATASETS_DIR / dataset_name
        self._dataset = None
        self._resume_process()

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def _resume_process(self):
        """ Resumes the process if there is a dataset already created. """
        if self.output_dir.exists():
            self._dataset = load_dataset(self.output_dir)
        else:
            self._dataset = None

    def save(self):
        """ Saves the dataset to the output directory. """
        self._dataset.save_to_disk(self.output_dir)

    def add_data(self, data: dict):
        """ Adds data to the dataset. """
        if self._dataset is None:
            self._dataset = Dataset.from_list([data])
        else:
            self._dataset = self._dataset.add_item(data)
