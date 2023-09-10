# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for json files."""
import os
from argparse import ArgumentParser, Namespace
from enum import Enum
from glob import glob
from typing import Dict, Iterable, Optional

import datasets as hf_datasets
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import os
import warnings
from typing import Dict, Iterable, Union, List

import datasets as hf_datasets
import numpy as np
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

import torch
from torch.nn.utils.rnn import pad_sequence

import imageio  # NOTE: I tried to use cv2.VideoCapture, but it didn't work well.
import json


def convert_time_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds

# NOTE: DO NOT USE THIS CLASS.
class NoConcatDataset(IterableDataset):
    """An IterableDataset that returns text samples for MDSWriter.

    Returns dicts of {'text': bytes}
    """

    def __init__(self, json_files: List[str], mp4_files: List[str]):
        self.json_files = json_files
        self.mp4_files = mp4_files

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for json_file, mpf_file in zip(self.json_files, self.mp4_files):
            if os.path.exists(json_file) is False:
                continue
            # print(sample)
            # convert to bytes to store in MDS binary format
            # yield {'text': sample['text'].encode('utf-8')}
            pass # TODO: implement this


class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is non-writeable
        # and torch does not support non-writeable tensors, so you get a scary warning and
        # if you do try to write to the tensor you get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(
        self,
        json_files: List[str],
        mp4_files: List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
    ):
        self.json_files = json_files
        self.mp4_files = mp4_files
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length - 10 # FIX THIS # TODO
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        print("eos token", self.eos_tokens)
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.')

        eos_text_provided = self.eos_text != ''
        bos_text_provided = self.bos_text != ''
        test_text = self.tokenizer('')
        if len(test_text['input_ids']) > 0 and (eos_text_provided or
                                                bos_text_provided):
            message = 'both eos and bos' if eos_text_provided and bos_text_provided else (
                'eos_text' if eos_text_provided else 'bos_text')
            warnings.warn(
                f'The provided tokenizer adds special tokens, but you also specified {message}. This may result '
                +
                'in duplicated special tokens. Please be sure this is what you intend.'
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        for json_file, mp4_file in zip(self.json_files, self.mp4_files):

            reader = imageio.get_reader(mp4_file)
            frames = []
            for frame in reader:
                frame_array = np.array(frame)
                frames.append(frame_array)

            # NOTE: To avoid a double loop of subtitle_data and reader for frames, store only the contents of subtitle_data in a numpy array and access the element later with np.where. 
            subtitle_texts = []
            subtitle_timestamps = []

            with open(json_file, 'r') as f:
                data = json.load(f)
                fps = reader.get_meta_data()['fps']
                try:
                    subtitle_data_list = data["yt_meta_dict"]["subtitles"]  # NOTE: This is not always available
                except KeyError:
                    continue

                for subtitle_data in subtitle_data_list:
                    subtitle_timestamp = {}
                    start_time = convert_time_to_seconds(subtitle_data["start"])  # Start time of the subtitle.
                    end_time = convert_time_to_seconds(subtitle_data["end"])  # End time of the subtitle.

                    subtitle_timestamp["start"] = start_time
                    subtitle_timestamp["end"] = end_time

                    subtitle_text_eoncoded = self.tokenizer(subtitle_data["lines"][0],
                                                            truncation=False,
                                                            padding=False)
                    subtitle_text = self.bos_tokens + subtitle_text_eoncoded['input_ids'] + self.eos_tokens
                    subtitle_texts.append(subtitle_text)
                    subtitle_timestamps.append(subtitle_timestamp)

            subtitle_data = {
                'subtitle': subtitle_texts,
                'subtitle_timestamp': subtitle_timestamps,
            }

            yield {
                'frame': np.asarray(frames, dtype=np.float32).transpose(0, 3, 1, 2),  # N, H, W, C -> N, C, H, W
                'subtitle_data': subtitle_data,
            }
    
class ConcatMode(Enum):
    NO_CONCAT = 'NO_CONCAT'
    CONCAT_TOKENS = 'CONCAT_TOKENS'


'''
python create_dataset.py \
  --path /p/fastdata/mmlaion/hummingbird/streaming/arxiv.jsonl \
  --out_root /p/fastdata/mmlaion/hummingbird/streaming/text/train --split train \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>' \
  --compression zstd
'''
def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing'
    )
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens')
    parser.add_argument('--split', type=str, default='train')  # NOTE: I assume this action is not used.

    parser.add_argument('--tokenizer', type=str, required=False, default=None)
    parser.add_argument('--bos_text', type=str, required=False, default=None)
    parser.add_argument('--eos_text', type=str, required=False, default=None)
    parser.add_argument('--no_wrap', default=False, action='store_true')

    parsed = parser.parse_args()

    if os.path.isdir(parsed.out_root) and len(
            set(os.listdir(parsed.out_root)).intersection(set(
                parsed.split))) > 0:
        raise ValueError(
            f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.'
        )

    # Make sure we have needed concat options
    if (parsed.concat_tokens is not None and
            isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
        parser.error(
            'When setting --concat_tokens, you must specify a --tokenizer')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


def build_hf_dataset(
    path: str,
    split: str,
    mode: ConcatMode,
    max_length: Optional[int] = None,
    bos_text: str = '',
    eos_text: str = '',
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase = None,
) -> IterableDataset:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        dataset_name (str): Dataset name
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns:
        An IterableDataset.
    """
    # NOTE: Do not use a single file.
    # if os.path.isdir(path):
    #     data_files = glob(f'{path}/*')
    # else:
    #     data_files = path
    assert os.path.isdir(path)

    # NOTE: I don't use huggingface datasets.
    # hf_dataset = hf_datasets.load_dataset('json',
    #                                       data_files=data_files,
    #                                       split=split)
    json_files = glob(os.path.join(path, "*.json"))
    mp4_files = glob(os.path.join(path, "*.mp4"))

    # Only use files that have both json and mp4 files.
    json_basenames = {os.path.splitext(os.path.basename(f))[0] for f in json_files}
    mp4_basenames = {os.path.splitext(os.path.basename(f))[0] for f in mp4_files}
    common_basenames = json_basenames & mp4_basenames
    json_files = [f for f in json_files if os.path.splitext(os.path.basename(f))[0] in common_basenames]
    mp4_files = [f for f in mp4_files if os.path.splitext(os.path.basename(f))[0] in common_basenames]

    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(json_files, mp4_files)
    else:
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                f'{tokenizer=} must be of type PreTrainedTokenizerBase')
        if max_length is None:
            raise ValueError(f'max_length must be set.')
        if bos_text + eos_text == '':
            test_tokens = tokenizer('test')
            if test_tokens['input_ids'][
                    0] != tokenizer.bos_token_id and test_tokens['input_ids'][
                        -1] != tokenizer.eos_token_id:
                tok_error_msg = 'This tokenizer does not insert an EOS nor BOS token. '
                tok_error_msg += 'Concatenating with this tokenizer will result in sequences being '
                tok_error_msg += 'attached without a separating token. Please use another tokenizer, '
                tok_error_msg += 'such as facebook/opt-125m, or specify EOS/BOS text with e.g. '
                tok_error_msg += '--bos_text=<|endoftext|>.'
                raise ValueError(tok_error_msg)
        dataset = ConcatTokensDataset(json_files=json_files,
                                      mp4_files=mp4_files,
                                      tokenizer=tokenizer,
                                      max_length=max_length,
                                      bos_text=bos_text,
                                      eos_text=eos_text,
                                      no_wrap=no_wrap)
    return dataset


def generate_samples(
        loader: DataLoader,
        truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        print(keys)
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}


def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    if args.concat_tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        # we will enforce length, so suppress warnings about sequences too long for the model
        tokenizer.model_max_length = int(1e30)
        columns = {'frame': 'ndarray', 'subtitle_data': 'json'}
    else:  # NOTE: DO NOT USE THIS OPTION.
        mode = ConcatMode.NO_CONCAT
        tokenizer = None
        columns = {'text': 'str'}

    # Get samples
    dataset = build_hf_dataset(path=args.path,
                               split=args.split,
                               mode=mode,
                               max_length=args.concat_tokens,
                               bos_text=args.bos_text,
                               eos_text=args.eos_text,
                               no_wrap=args.no_wrap,
                               tokenizer=tokenizer)

    print('here')

    # Write samples
    print(f'Converting to MDS format...')
    print(
        f'Note that the progress bar is based on the dataset length before tokenization.'
    )
    print(f'It will finish at a value below 100% if tokenizing')
    with MDSWriter(columns=columns,
                   out=os.path.join(args.out_root),
                   compression=args.compression) as out:
        for sample in tqdm(dataset):
            out.write(sample)


if __name__ == '__main__':
    main(parse_args())


'''
python create_dataset.py   --path /p/fastdata/mmlaion/hummingbird/streaming/arxiv.jsonl   --out_root /p/fastdata/mmlaion/hummingbird/streaming/interleaved/train --split train   --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'   --compression zstd'''