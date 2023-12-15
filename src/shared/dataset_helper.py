from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


class DatasetHelper:

    def __init__(self, dataset, tokenizer, batch_size):
        self.dataset = self.create_data_splits(dataset)
        self.data_collator = DataCollatorWithPadding(
            tokenizer, padding="longest", return_tensors="pt"
        )
        self.batch_size = batch_size

    def create_data_splits(self, dataset):
        print("First dataset entry: ", dataset[0])
        # split twice and combine
        train_devtest = dataset.train_test_split(
            shuffle=True, seed=200, test_size=0.3
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

    # TODO: Replace dataset with key as soon as tokenization is implemented here
    def load_tokenized_dataset(self, tokenized_dataset, shuffle=False):
        return DataLoader(tokenized_dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.data_collator)

    def apply_tokenization(self, tokenizer_function, batched=False):
        return self.dataset.map(tokenizer_function, remove_columns=self.dataset["train"].column_names, batched=batched)
