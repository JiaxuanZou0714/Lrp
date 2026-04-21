import argparse
import json
import os
import shutil
import time

import datasets


RESUME_DIR_NAME = ".prepare_data_resume"
MANIFEST_FILE_NAME = "manifest.json"


def _manifest_path(split_resume_dir):
    return os.path.join(split_resume_dir, MANIFEST_FILE_NAME)


def _atomic_write_json(obj, file_path):
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)
    os.replace(tmp_path, file_path)


def _chunk_index(chunk_name):
    try:
        return int(chunk_name.split("_", 1)[1])
    except Exception:
        return -1


def _list_chunk_dirs(split_resume_dir):
    if not os.path.isdir(split_resume_dir):
        return []
    chunk_dirs = []
    for name in os.listdir(split_resume_dir):
        full_path = os.path.join(split_resume_dir, name)
        if name.startswith("chunk_") and os.path.isdir(full_path):
            chunk_dirs.append(name)
    return sorted(chunk_dirs, key=_chunk_index)


def _load_or_init_manifest(split_name, split_resume_dir, target_examples, chunk_size):
    os.makedirs(split_resume_dir, exist_ok=True)
    manifest_path = _manifest_path(split_resume_dir)
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {
            "split_name": split_name,
            "target_examples": target_examples,
            "chunk_size": chunk_size,
            "completed_examples": 0,
            "next_chunk_idx": 0,
            "chunks": [],
        }

    if manifest.get("split_name") != split_name:
        raise ValueError(
            f"Manifest split mismatch for {split_name}: got {manifest.get('split_name')}"
        )

    # Allow extending target count across runs.
    manifest["target_examples"] = target_examples
    manifest["chunk_size"] = chunk_size

    # Reconcile chunks that exist on disk but were not recorded (e.g. crash after save).
    recorded = {chunk["name"] for chunk in manifest.get("chunks", [])}
    for chunk_name in _list_chunk_dirs(split_resume_dir):
        if chunk_name in recorded:
            continue
        chunk_rows = datasets.load_from_disk(os.path.join(split_resume_dir, chunk_name)).num_rows
        manifest.setdefault("chunks", []).append({"name": chunk_name, "rows": int(chunk_rows)})

    manifest["chunks"] = sorted(manifest.get("chunks", []), key=lambda c: _chunk_index(c["name"]))
    manifest["completed_examples"] = int(sum(chunk["rows"] for chunk in manifest["chunks"]))

    max_chunk_idx = -1
    for chunk in manifest["chunks"]:
        max_chunk_idx = max(max_chunk_idx, _chunk_index(chunk["name"]))
    manifest["next_chunk_idx"] = max_chunk_idx + 1

    if manifest["completed_examples"] > target_examples:
        raise ValueError(
            f"Resume state for {split_name} already has {manifest['completed_examples']} examples, "
            f"which is greater than requested {target_examples}. Use --overwrite to restart."
        )

    _atomic_write_json(manifest, manifest_path)
    return manifest


def _stream_from(split_name, start_index):
    data_stream = datasets.load_dataset("allenai/c4", "en", split=split_name, streaming=True)
    if start_index > 0 and hasattr(data_stream, "skip"):
        data_stream = data_stream.skip(start_index)
        start_index = 0

    iterator = iter(data_stream)
    while start_index > 0:
        try:
            next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"Could not skip to index for split={split_name}, requested skip={start_index}"
            ) from exc
        start_index -= 1

    return iterator


def _finalize_split(split_name, split_dir, split_resume_dir, manifest, keep_resume_chunks):
    chunk_paths = [os.path.join(split_resume_dir, chunk["name"]) for chunk in manifest["chunks"]]
    if not chunk_paths:
        raise RuntimeError(f"No chunk data found for split {split_name}; cannot finalize")

    print(f"Combining {len(chunk_paths)} chunk(s) for split={split_name}...")
    chunk_datasets = [datasets.load_from_disk(path) for path in chunk_paths]
    if len(chunk_datasets) == 1:
        merged = chunk_datasets[0]
    else:
        merged = datasets.concatenate_datasets(chunk_datasets)

    if os.path.isdir(split_dir):
        shutil.rmtree(split_dir)
    merged.save_to_disk(split_dir)

    final_rows = datasets.load_from_disk(split_dir).num_rows
    if final_rows != manifest["target_examples"]:
        raise RuntimeError(
            f"Finalized split {split_name} has {final_rows} rows, expected {manifest['target_examples']}"
        )

    print(f"Finalized split={split_name} with {final_rows} examples at {split_dir}")

    if not keep_resume_chunks:
        shutil.rmtree(split_resume_dir, ignore_errors=True)


def _take_items_with_retries(
    split_name,
    stream_iter,
    start_index,
    take_n,
    max_stream_errors,
    retry_initial_wait,
    retry_max_wait,
):
    items = []
    stream_errors = 0
    while len(items) < take_n:
        try:
            items.append(next(stream_iter))
        except StopIteration:
            break
        except Exception as exc:
            stream_errors += 1
            current_index = start_index + len(items)
            if stream_errors > max_stream_errors:
                raise RuntimeError(
                    f"Too many stream errors while reading split={split_name} at index={current_index}; "
                    f"max_stream_errors={max_stream_errors}"
                ) from exc

            wait_seconds = min(retry_initial_wait * (2 ** (stream_errors - 1)), retry_max_wait)
            print(
                f"Warning: stream read error at split={split_name}, index={current_index}: {type(exc).__name__}: {exc}"
            )
            print(
                f"Reconnecting stream and retrying in {wait_seconds}s "
                f"({stream_errors}/{max_stream_errors})"
            )
            time.sleep(wait_seconds)
            stream_iter = _stream_from(split_name, current_index)

    return items, stream_iter


def _prepare_split(
    split_name,
    target_examples,
    save_path,
    chunk_size,
    overwrite=False,
    keep_resume_chunks=False,
    max_stream_errors=20,
    retry_initial_wait=2,
    retry_max_wait=60,
):
    split_dir = os.path.join(save_path, split_name)
    split_resume_dir = os.path.join(save_path, RESUME_DIR_NAME, split_name)

    if overwrite:
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)
        if os.path.isdir(split_resume_dir):
            shutil.rmtree(split_resume_dir)

    manifest = _load_or_init_manifest(
        split_name=split_name,
        split_resume_dir=split_resume_dir,
        target_examples=target_examples,
        chunk_size=chunk_size,
    )

    # Seed resume chunks from existing finalized split so we can extend without re-downloading.
    if os.path.isdir(split_dir):
        existing_rows = datasets.load_from_disk(split_dir).num_rows
        if existing_rows == target_examples:
            print(f"Split {split_name} already complete: {existing_rows}/{target_examples}. Skipping.")
            if not keep_resume_chunks and os.path.isdir(split_resume_dir):
                shutil.rmtree(split_resume_dir, ignore_errors=True)
            return

        if existing_rows > target_examples:
            print(
                f"Split {split_name} already has {existing_rows} examples which is greater than requested "
                f"{target_examples}. Keeping existing split and skipping."
            )
            return

        if manifest["completed_examples"] == 0:
            seed_chunk_name = f"chunk_{manifest['next_chunk_idx']:06d}"
            seed_chunk_dir = os.path.join(split_resume_dir, seed_chunk_name)
            print(
                f"Seeding resume state for split={split_name} from existing {existing_rows} examples "
                f"in {split_dir}"
            )
            shutil.move(split_dir, seed_chunk_dir)
            manifest["chunks"].append({"name": seed_chunk_name, "rows": int(existing_rows)})
            manifest["chunks"] = sorted(manifest["chunks"], key=lambda c: _chunk_index(c["name"]))
            manifest["completed_examples"] = int(sum(chunk["rows"] for chunk in manifest["chunks"]))
            manifest["next_chunk_idx"] = _chunk_index(seed_chunk_name) + 1
            _atomic_write_json(manifest, _manifest_path(split_resume_dir))
        else:
            print(
                f"Found existing split dir {split_dir} while resume state is already present; "
                f"resume chunks take precedence."
            )

    completed = manifest["completed_examples"]
    if completed == target_examples:
        print(f"All chunks already downloaded for split={split_name}. Finalizing...")
        _finalize_split(
            split_name=split_name,
            split_dir=split_dir,
            split_resume_dir=split_resume_dir,
            manifest=manifest,
            keep_resume_chunks=keep_resume_chunks,
        )
        return

    print(f"Resuming split={split_name}: completed={completed}, target={target_examples}, chunk_size={chunk_size}")
    stream_iter = _stream_from(split_name, completed)

    while completed < target_examples:
        remaining = target_examples - completed
        take_n = min(chunk_size, remaining)
        chunk_name = f"chunk_{manifest['next_chunk_idx']:06d}"
        chunk_tmp_dir = os.path.join(split_resume_dir, f"{chunk_name}.tmp")
        chunk_final_dir = os.path.join(split_resume_dir, chunk_name)

        if os.path.isdir(chunk_tmp_dir):
            shutil.rmtree(chunk_tmp_dir)

        chunk_items, stream_iter = _take_items_with_retries(
            split_name=split_name,
            stream_iter=stream_iter,
            start_index=completed,
            take_n=take_n,
            max_stream_errors=max_stream_errors,
            retry_initial_wait=retry_initial_wait,
            retry_max_wait=retry_max_wait,
        )
        chunk_rows = len(chunk_items)
        if chunk_rows == 0:
            raise RuntimeError(
                f"Streaming source ended unexpectedly for split={split_name} at {completed}/{target_examples}"
            )

        chunk_dataset = datasets.Dataset.from_list(chunk_items)

        chunk_dataset.save_to_disk(chunk_tmp_dir)
        if os.path.isdir(chunk_final_dir):
            shutil.rmtree(chunk_final_dir)
        shutil.move(chunk_tmp_dir, chunk_final_dir)

        manifest["chunks"].append({"name": chunk_name, "rows": chunk_rows})
        manifest["completed_examples"] += chunk_rows
        manifest["next_chunk_idx"] += 1
        _atomic_write_json(manifest, _manifest_path(split_resume_dir))

        completed = manifest["completed_examples"]
        print(f"Downloaded chunk {chunk_name}: +{chunk_rows}, progress={completed}/{target_examples}")

    _finalize_split(
        split_name=split_name,
        split_dir=split_dir,
        split_resume_dir=split_resume_dir,
        manifest=manifest,
        keep_resume_chunks=keep_resume_chunks,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_examples", type=int, default=5200000)
    parser.add_argument("--val_examples", type=int, default=40000)
    parser.add_argument("--save_path", type=str, default="./c4_local")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="How many examples to download per resumable chunk",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing split and resume state before downloading",
    )
    parser.add_argument(
        "--keep_resume_chunks",
        action="store_true",
        help="Keep intermediate chunk data after finalizing split",
    )
    parser.add_argument(
        "--max_stream_errors",
        type=int,
        default=20,
        help="Maximum transient stream errors tolerated before aborting",
    )
    parser.add_argument(
        "--retry_initial_wait",
        type=int,
        default=2,
        help="Initial seconds to wait before reconnecting stream",
    )
    parser.add_argument(
        "--retry_max_wait",
        type=int,
        default=60,
        help="Maximum seconds for exponential reconnect backoff",
    )
    parser.add_argument(
        "--hub_download_timeout",
        type=int,
        default=60,
        help="Set HF_HUB_DOWNLOAD_TIMEOUT if not already defined",
    )
    args = parser.parse_args()

    if args.train_examples <= 0 or args.val_examples <= 0:
        raise ValueError("train_examples and val_examples must both be positive")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if args.max_stream_errors <= 0:
        raise ValueError("max_stream_errors must be positive")
    if args.retry_initial_wait <= 0 or args.retry_max_wait <= 0:
        raise ValueError("retry_initial_wait and retry_max_wait must be positive")
    if args.retry_initial_wait > args.retry_max_wait:
        raise ValueError("retry_initial_wait must be <= retry_max_wait")
    if args.hub_download_timeout <= 0:
        raise ValueError("hub_download_timeout must be positive")

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(args.hub_download_timeout))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(min(args.hub_download_timeout, 30)))
    print(
        "Using hub settings: "
        f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://huggingface.co')} "
        f"HF_HUB_DOWNLOAD_TIMEOUT={os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')}"
    )

    os.makedirs(args.save_path, exist_ok=True)

    print(f"Preparing train split with target={args.train_examples}...")
    _prepare_split(
        split_name="train",
        target_examples=args.train_examples,
        save_path=args.save_path,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        keep_resume_chunks=args.keep_resume_chunks,
        max_stream_errors=args.max_stream_errors,
        retry_initial_wait=args.retry_initial_wait,
        retry_max_wait=args.retry_max_wait,
    )

    print(f"Preparing validation split with target={args.val_examples}...")
    _prepare_split(
        split_name="validation",
        target_examples=args.val_examples,
        save_path=args.save_path,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        keep_resume_chunks=args.keep_resume_chunks,
        max_stream_errors=args.max_stream_errors,
        retry_initial_wait=args.retry_initial_wait,
        retry_max_wait=args.retry_max_wait,
    )

    resume_root = os.path.join(args.save_path, RESUME_DIR_NAME)
    if os.path.isdir(resume_root) and not os.listdir(resume_root):
        os.rmdir(resume_root)

    print(f"Dataset successfully saved to {args.save_path}")

if __name__ == "__main__":
    main()
