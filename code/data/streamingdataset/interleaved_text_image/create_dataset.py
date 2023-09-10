# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for json files."""
import os
from argparse import ArgumentParser, Namespace
from enum import Enum
from glob import glob
from typing import Dict, Iterable, Optional

from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import os
import warnings
from typing import Dict, Iterable, Union, List

import numpy as np
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

# import torch
from torch.nn.utils.rnn import pad_sequence

import imageio  # NOTE: I tried to use cv2.VideoCapture, but it didn't work well.
import json


def convert_time_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds

# NOTE: DO NOT USE THIS CLASS.
# class NoConcatDataset(IterableDataset):
#     """An IterableDataset that returns text samples for MDSWriter.

#     Returns dicts of {'text': bytes}
#     """

#     def __init__(self, json_files: List[str], mp4_files: List[str]):
#         self.json_files = json_files
#         self.mp4_files = mp4_files

#     def __iter__(self) -> Iterable[Dict[str, bytes]]:
#         for json_file, mpf_file in zip(self.json_files, self.mp4_files):
#             if os.path.exists(json_file) is False:
#                 continue
#             # print(sample)
#             # convert to bytes to store in MDS binary format
#             # yield {'text': sample['text'].encode('utf-8')}
#             pass


class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder')
    ```
    """

    def __init__(
        self,
        json_files: List[str],
        mp4_files: List[str],
        tokenizer: PreTrainedTokenizerBase,
        bos_text: str,
        eos_text: str,
    ):
        self.json_files = json_files
        self.mp4_files = mp4_files
        self.tokenizer = tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.bos_text = bos_text
        self.eos_text = eos_text

        self.bos_tokens = self.tokenizer(self.bos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f'You specified --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.')

        self.eos_tokens = self.tokenizer(self.eos_text,
                                         truncation=False,
                                         padding=False,
                                         add_special_tokens=False)['input_ids']
        print("eos token", self.eos_tokens)
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --eos_text, but your EOS text is not tokenizing to one token\
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

# NOTE: Always concat tokens.
# class ConcatMode(Enum):
#     NO_CONCAT = 'NO_CONCAT'
#     CONCAT_TOKENS = 'CONCAT_TOKENS'


'''
python create_dataset.py \
  --path /root/projects/rl-nlp/videos/dataset_downsampled/00000/ \
  --out_root /root/projects/rl-nlp/videos/data_downsampled/mds-not-split/00000 \
  --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>' \
  --compression zstd
'''
def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, concatenating and tokenizing'
    )
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)

    # NOTE: Always concatenate tokens.
    # group = parser.add_mutually_exclusive_group(required=False)
    # group.add_argument(
    #     '--concat_tokens',
    #     type=int,
    #     help='Convert text to tokens and concatenate up to this many tokens')

    # parser.add_argument('--split', type=str, default='train')  # NOTE: HD-VILA-100M does not have a training/validation split.

    parser.add_argument('--tokenizer', type=str, required=True, default=None)  # NOTE: Always tokenize the text.
    parser.add_argument('--bos_text', type=str, required=False, default=None)
    parser.add_argument('--eos_text', type=str, required=False, default=None)
    # parser.add_argument('--no_wrap', default=False, action='store_true')  # NOTE: DO NOT USE THIS OPTION.

    parsed = parser.parse_args()

    # NOTE: No splitting.
    # if os.path.isdir(parsed.out_root) and len(
    #         set(os.listdir(parsed.out_root)).intersection(set(
    #             parsed.split))) > 0:
    #     raise ValueError(
    #         f'--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.'
    #     )
    if os.path.exists(parsed.out_root + '/*'):
        raise ValueError(
            f'Files exist in --out_root={parsed.out_root}.'
        )

    # NOTE: Always concatenate tokens.
    # if (parsed.concat_tokens is not None and
    #         isinstance(parsed.concat_tokens, int) and parsed.tokenizer is None):
    #     parser.error(
    #         'When setting --concat_tokens, you must specify a --tokenizer')

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ''
    if parsed.eos_text is None:
        parsed.eos_text = ''
    return parsed


def build_dataset(
    path: str,
    bos_text: str = '',
    eos_text: str = '',
    tokenizer: PreTrainedTokenizerBase = None,
) -> IterableDataset:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        path (str): Path to the dataset
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        tokenizer (PreTrainedTokenizerBase): tokenizer to use

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

    # NOTE: Always concatenate tokens.
    # if mode == ConcatMode.NO_CONCAT:
    #     dataset = NoConcatDataset(json_files, mp4_files)
    # else:
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError(
            f'{tokenizer=} must be of type PreTrainedTokenizerBase')
    # NOTE: I don't use max_length.
    # if max_length is None:
    #     raise ValueError(f'max_length must be set.')
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
                                    bos_text=bos_text,
                                    eos_text=eos_text)
    return dataset


def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    # NOTE: Always concat tokens.
    # if args.concat_tokens is not None:
    # mode = ConcatMode.CONCAT_TOKENS
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # we will enforce length, so suppress warnings about sequences too long for the model
    tokenizer.model_max_length = int(1e30)
    columns = {'frame': 'ndarray', 'subtitle_data': 'json'}

    # NOTE: Always concat tokens.
    # else:
    #     mode = ConcatMode.NO_CONCAT
    #     tokenizer = None
    #     columns = {'text': 'str'}

    # Get samples
    dataset = build_dataset(path=args.path,
                            bos_text=args.bos_text,
                            eos_text=args.eos_text,
                            tokenizer=tokenizer)

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
python create_dataset.py   --path /root/projects/rl-nlp/videos/dataset_downsampled/00000/   --out_root /root/projects/rl-nlp/videos/data_downsampled/mds-not-split/00000/ --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'   --compression zstd'''