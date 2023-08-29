import os
import json
import imageio  # NOTE: I tried to use cv2.VideoCapture, but it didn't work well.
import numpy as np
from streaming import MDSWriter
from tqdm import tqdm

# NOTE: `dataset_downsampled` contains the dataset of 1fps videos.
BASE_DIR = "/root/projects/rl-nlp/videos/dataset_downsampled/00000/"  # NOTE: video2dataset creates more directories other than 00000. I used only 00000 because my machine doesn't have enough space.

def convert_time_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds


SAVE_PATH = "/root/projects/rl-nlp/videos/data_downsampled/mds/00000"
os.makedirs(SAVE_PATH, exist_ok=True)

COLUMNS = {
    'frame': 'ndarray',
    'subtitle': 'ndarray'
}
COMPRESSION = 'zstd'
HASHES = 'sha1', 'xxh64'

with MDSWriter(out=SAVE_PATH, columns=COLUMNS, compression=COMPRESSION, hashes=HASHES) as save_file:

    for video_filename in tqdm(os.listdir(BASE_DIR)):
        if not video_filename.endswith(".mp4"):
            continue

        file_basename = os.path.splitext(video_filename)[0]

        # Each video is expected to have a pair of video file and json file.
        video_file_path = os.path.join(BASE_DIR, video_filename)
        json_file_path = os.path.join(BASE_DIR, f"{file_basename}.json")

        if os.path.exists(json_file_path):
            pass
        else:
            continue  # NOTE: Some videos miss the json file

        reader = imageio.get_reader(video_file_path)

        frames = []
        subtitles = []

        with open(json_file_path, 'r') as f:
            data = json.load(f)
            fps = reader.get_meta_data()['fps']

            # fps = data["yt_meta_dict"]["info"]["fps"]  # NOTE: This is not always available
            try:
                subtitle_data = data["yt_meta_dict"]["subtitles"]  # NOTE: This is not always available
            except KeyError:
                continue

            for i, subtitle in enumerate(subtitle_data):
                start_time = convert_time_to_seconds(subtitle["start"])  # Start time of the subtitle.
                end_time = convert_time_to_seconds(subtitle["end"])  # End time of the subtitle.

                while start_time < end_time:
                    chunk_start_time = start_time
                    chunk_end_time = start_time + 1.0  #  Chunk frames every 2 seconds: chunk = [frame_0, frame_1]

                    if chunk_end_time > end_time:
                        chunk_end_time = end_time

                    chunk_frame = []
                    chunk_subtitle = []
                    while chunk_start_time < chunk_end_time:

                        frame_idx = int(chunk_start_time * fps)

                        try:
                            frame = reader.get_data(frame_idx)
                            frame_array = np.array(frame)
                            chunk_frame.append(frame_array)
                            chunk_subtitle.append(np.frombuffer(subtitle["lines"][0].encode('utf-8'), dtype=np.uint8))
                        except IndexError:
                            break  # NOTE: I sometimes get IndexError even when frame_idx is less than the number of frames. I don't know why.

                        if frame is None:
                            continue

                        chunk_start_time += 1.0  # NOTE: 1fps

                    if len(chunk_frame) > 0 and len(chunk_subtitle) > 0:
                        frames.append(chunk_frame)
                        subtitles.append(chunk_subtitle)

                    start_time = chunk_end_time

        if len(frames) == 0 or len(subtitles) == 0:
            continue

        try:
            max_length = 0
            for sub_chunk in subtitles:
                for sub in sub_chunk:
                    max_length = max(max_length, len(sub))
        except ValueError:
            print('max() arg is an empty sequence')  # NOTE: If all intervals of subtitles are less than 1 second, we get an empty sequence.
            continue
        
        padded_subtitles = []
        for sub_chunk in subtitles:
            padded_sub_chunk = []
            for sub in sub_chunk:
                padded_sub_chunk.append(np.pad(sub, (0, max_length - len(sub))))  # NOTE: Pad each subtitle to the same length
            padded_subtitles.append(padded_sub_chunk)

        # padded_subtitles = [np.pad(sub, (0, MAX_LENGTH - len(sub))) for sub in subtitles]

        video_data = {
            'frame': np.asarray(frames, dtype=np.float32),
            'subtitle': np.asarray(padded_subtitles)
        }

        save_file.write(video_data)
