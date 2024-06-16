from datasets import DatasetDict, Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

import constants


class DatasetHelper:

    def __init__(self, raw_dataset, tokenizer, batch_size=16):
        self.dataset = self.create_data_splits(raw_dataset)
        self.cross_validated_datasets = self.create_cross_validation_dsdicts(raw_dataset)
        self.data_collator = DataCollatorWithPadding(
            tokenizer, padding="longest", return_tensors="pt"
        )
        self.batch_size = batch_size

    def create_data_splits(self, dataset):
        # split twice and combine
        train_devtest = dataset.train_test_split(
            shuffle=True, seed=200, test_size=0.2
        )
        hf_dev_test = train_devtest["test"].train_test_split(
            shuffle=True, seed=200, test_size=0.50
        )
        hf_dataset = DatasetDict(
            {
                "train": train_devtest["train"],
                "test": hf_dev_test["test"],
                "validation": hf_dev_test["train"],
            }
        )
        return hf_dataset

    def create_cross_validation_dsdicts(self, dataset):
        shuffled_dataset = dataset.shuffle(seed=200)
        shuffled_dataset.to_parquet(constants.SEQ_LABELLING_DATASET_PATH)
        # https://huggingface.co/docs/datasets/v2.12.0/en/loading#slice-splits
        # https://scikit-learn.org/stable/modules/cross_validation.html
        # Fixed validation split (e.g., the first 10%)
        fixed_val_split = "train[0%:10%]"
        # Generate splits
        splits = range(10, 100, 10)
        validation_set = load_dataset("parquet", data_files={"train": [constants.SEQ_LABELLING_DATASET_PATH]},
                                              split=fixed_val_split)
        k_fold_test_sets = load_dataset("parquet", data_files={"train": [constants.SEQ_LABELLING_DATASET_PATH]},
                                        split=[f"train[{(k)}%:{(k+10)}%]" for k in splits])
        k_fold_training_sets = load_dataset("parquet", data_files={"train": [constants.SEQ_LABELLING_DATASET_PATH]},
                                            split=[f"train[10%:{(k)}%]+train[{(k+10)}%:]" for k in splits])

        dataset_dicts = []
        for train_set, test_set in zip(k_fold_training_sets, k_fold_test_sets):

            hf_dataset = DatasetDict(
                {
                    "train": train_set,
                    "test": test_set,
                    "validation": validation_set,
                }
            )
            dataset_dicts.append(hf_dataset)

        return dataset_dicts


    @staticmethod
    def create_data_splits_qa(train_dictlist, eval_test_dictlist):
        train_dataset = Dataset.from_list(train_dictlist)
        eval_test_dataset = Dataset.from_list(eval_test_dictlist)
        eval_test_split_dataset = eval_test_dataset.train_test_split(
            shuffle=True, seed=200, test_size=0.5
        )
        hf_dataset = DatasetDict(
            {
                "train": train_dataset,
                "test": eval_test_split_dataset["test"],
                "validation": eval_test_split_dataset["train"],
            }
        )
        return hf_dataset

    def load_tokenized_dataset(self, tokenized_dataset, shuffle=False):
        return DataLoader(tokenized_dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.data_collator)

    def apply_tokenization(self, tokenizer_function, batched=False):
        return self.dataset.map(tokenizer_function, remove_columns=self.dataset["train"].column_names, batched=batched)
