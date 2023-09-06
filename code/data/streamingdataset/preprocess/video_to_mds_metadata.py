import os
import json
import imageio  # NOTE: I tried to use cv2.VideoCapture, but it didn't work well.
import numpy as np
from streaming import MDSWriter
from tqdm import tqdm
import json

BASE_DIR = "/root/projects/rl-nlp/videos/dataset_downsampled/00000/"  # NOTE: video2dataset creates more directories other than 00000. I used only 00000 because my machine doesn't have enough space.

def convert_time_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds


SAVE_PATH = "/root/projects/rl-nlp/videos/data_downsampled/mds-not-split/00000"
os.makedirs(SAVE_PATH, exist_ok=True)

COLUMNS = {
    'frame': 'ndarray',
    'subtitle_data': 'json'
}
COMPRESSION = 'zstd'
HASHES = 'sha1', 'xxh64'

with MDSWriter(out=SAVE_PATH, columns=COLUMNS, compression=COMPRESSION, hashes=HASHES) as save_file:        

    for video_filename in tqdm(os.listdir(BASE_DIR)[:10]):  # NOTE: I use only 10 videos for the test. I checked that the code works well with the full number of videos in 00000.
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
        for frame_number, frame in enumerate(reader):
            frame_array = np.array(frame)
            frames.append(frame_array)

        # NOTE: To avoid a double loop of subtitle_data and reader for frames, store only the contents of subtitle_data in a numpy array and access the element later with np.where. 
        subtitle_texts = []
        subtitle_timestamps = []

        with open(json_file_path, 'r') as f:
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

                start_frame_number = int(start_time * fps)
                end_frame_number = int(end_time * fps)

                subtitle_timestamp["start"] = start_time
                subtitle_timestamp["end"] = end_time

                subtitle_texts.append(subtitle_data["lines"][0])
                subtitle_timestamps.append(subtitle_timestamp)

        subtitle_data = {
            'subtitle': subtitle_texts,
            'subtitle_timestamp': subtitle_timestamps,
        }

        video_data = {
            'frame': np.asarray(frames, dtype=np.float32),
            'subtitle_data': json.dumps(subtitle_data),
        }

        save_file.write(video_data)
