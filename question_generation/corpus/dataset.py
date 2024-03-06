""" Script to load and process the corpus that will be used in the QA generation process. """
import shutil
import pandas as pd
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
        # print first 10 examples
        print(self._dataset[self.split][:10])

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

    def get_rows(self) -> list[dict]:
        """ Returns the rows of the dataset. """
        return self._dataset[self.split].to_list()

    def get_titles(self) -> list[str]:
        """ Returns the titles of the dataset. """
        return self._dataset[self.split]['title']


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

    @property
    def dataset_length(self) -> int:
        return len(self._dataset) if self._dataset is not None else 0

    def get_titles(self) -> list[str]:
        """ Returns the titles of the dataset. """
        return self._dataset['title'].to_list() if self._dataset is not None else []

    def get_rows(self) -> list[dict]:
        """ Returns the rows of the dataset. """
        return self._dataset.to_dict('records') if self._dataset is not None else []
    
    def get_texts(self) -> list[str]:
        """ Returns the texts of the dataset. """
        return self._dataset['text'].to_list() if self._dataset is not None else []
        

    def _resume_process(self):
        """ Resumes the process if there is a dataset already created. """
        output_path = Path(str(self.output_dir) + '.csv')
        if output_path.exists():
            self._dataset = pd.read_csv(
                output_path,on_bad_lines='skip', lineterminator='\n')
            
            print(f"Resuming process with {len(self._dataset)} rows.")
        else:
            self._dataset = None

    def save(self):
        """ Saves the dataset to the output directory. """
        self._dataset.to_csv(
            Path(str(self.output_dir) + '.csv'), index=False)

    def add_data(self, data: dict):
        """ Adds data to the dataset. """
        if self._dataset is None:
            self._dataset = pd.DataFrame([data])
        else:
            self._dataset = pd.concat([self._dataset, pd.DataFrame([data])])

    def get_hf_dataset(self) -> Dataset:
        """ Returns the dataset in Hugging Face format. """
        return Dataset.from_pandas(self._dataset)
         
