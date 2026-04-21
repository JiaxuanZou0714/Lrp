import torch
from torch.utils.data import IterableDataset


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        # Upstream dataset partitioning (rank/world_size + DataLoader internals) already
        # handles worker/rank sharding. Avoid a second manual split here.
        iter_data = iter(self.data)

        text_batch = []
        for example in iter_data:
            text_batch.append(example["text"])

            if len(text_batch) == self.batch_size:
                yield self._tokenize_batch(text_batch)
                text_batch = []

        if text_batch:
            yield self._tokenize_batch(text_batch)

    def _tokenize_batch(self, texts):
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }


class TokenizedIterableDataset(IterableDataset):
    """Batches pre-tokenized samples already containing input_ids/attention_mask."""

    def __init__(self, data, batch_size):
        super().__init__()
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        iter_data = iter(self.data)
        batch = []

        for example in iter_data:
            batch.append(example)
            if len(batch) == self.batch_size:
                yield self._collate_batch(batch)
                batch = []

        if batch:
            yield self._collate_batch(batch)

    def _collate_batch(self, batch):
        return {
            "input_ids": torch.tensor([example["input_ids"] for example in batch], dtype=torch.long),
            "attention_mask": torch.tensor([example["attention_mask"] for example in batch], dtype=torch.long),
        }
