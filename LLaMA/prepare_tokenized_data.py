import argparse
import os
import shutil

import datasets
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-tokenize local C4 dataset for faster training")
    parser.add_argument(
        "--input_data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "c4_local"),
        help="Directory containing raw local dataset with train/validation splits",
    )
    parser.add_argument(
        "--output_data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "c4_tokenized_t5_base"),
        help="Directory to save tokenized train/validation splits",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="t5-base",
        help="Tokenizer name on HuggingFace Hub or local tokenizer directory",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_proc", type=int, default=8, help="Number of map workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output splits")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load tokenizer only from local files/cache",
    )
    return parser.parse_args()


def build_tokenizer(tokenizer_name_or_path, max_length, local_files_only=False):
    tokenizer_kwargs = {"model_max_length": max_length}
    if local_files_only:
        tokenizer_kwargs["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_split(split_name, input_data_dir, output_data_dir, tokenizer, max_length, num_proc, overwrite):
    src_split_dir = os.path.join(input_data_dir, split_name)
    dst_split_dir = os.path.join(output_data_dir, split_name)

    if not os.path.isdir(src_split_dir):
        raise FileNotFoundError(f"Input split not found: {src_split_dir}")

    if os.path.exists(dst_split_dir):
        if not overwrite:
            raise FileExistsError(
                f"Output split already exists: {dst_split_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(dst_split_dir)

    print(f"Loading split {split_name} from {src_split_dir}")
    dataset = datasets.load_from_disk(src_split_dir)

    if "text" not in dataset.column_names:
        raise ValueError(f"Split {split_name} does not contain a text column")

    remove_columns = list(dataset.column_names)

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    map_kwargs = {
        "batched": True,
        "remove_columns": remove_columns,
        "desc": f"Tokenizing {split_name}",
    }
    if num_proc is not None and num_proc > 1:
        map_kwargs["num_proc"] = num_proc

    tokenized_dataset = dataset.map(tokenize_batch, **map_kwargs)
    tokenized_dataset.save_to_disk(dst_split_dir)
    print(f"Saved tokenized split {split_name} to {dst_split_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_data_dir, exist_ok=True)

    tokenizer = build_tokenizer(
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )

    tokenize_split(
        split_name="train",
        input_data_dir=args.input_data_dir,
        output_data_dir=args.output_data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_proc=args.num_proc,
        overwrite=args.overwrite,
    )
    tokenize_split(
        split_name="validation",
        input_data_dir=args.input_data_dir,
        output_data_dir=args.output_data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_proc=args.num_proc,
        overwrite=args.overwrite,
    )

    print(f"Tokenized dataset saved under {args.output_data_dir}")


if __name__ == "__main__":
    main()
