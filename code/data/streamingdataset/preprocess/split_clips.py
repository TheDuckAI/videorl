import os
import json
import imageio  # NOTE: I tried to use cv2.VideoCapture, but it didn't work well.
import numpy as np
from streaming import MDSWriter
from tqdm import tqdm

BASE_DIR = "/root/projects/rl-nlp/videos/dataset/00000/"  # NOTE: video2dataset creates more directories other than 00000. I used only 00000 because my machine doesn't have enough space.

def convert_time_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds

videos = []

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

            while start_time < end_time:  # Cut videos between start_time and end_time into 1 second chunks.
                chunk_end_time = start_time + 1.0  #  1.0 second chunks
                if chunk_end_time > end_time:
                    chunk_end_time = end_time

                frame_idx = int(start_time * fps)

                try: 
                    frame = reader.get_data(frame_idx)
                except IndexError:
                    break  # NOTE: I sometimes get IndexError even when frame_idx is less than the number of frames. I don't know why.

                if frame is None:
                    break
                frame_array = np.array(frame)

                frames.append(frame_array)
                subtitles.append(np.frombuffer(subtitle["lines"][0].encode('utf-8'), dtype=np.uint8))

                start_time = chunk_end_time

    try:
        MAX_LENGTH = max([len(sub) for sub in subtitles])
    except ValueError:
        print('max() arg is an empty sequence')  # NOTE: If all intervals of subtitles are less than 1 second, we get an empty sequence.
        continue

    padded_subtitles = [np.pad(sub, (0, MAX_LENGTH - len(sub))) for sub in subtitles]  # NOTE: Pad each subtitle to the same length

    video_data = {
        'frame': np.asarray(frames, dtype=np.float32),
        'subtitle': np.asarray(padded_subtitles)
    }

    videos.append(video_data)

SAVE_PATH = "/root/projects/rl-nlp/videos/data/mds/00000"
os.makedirs(SAVE_PATH, exist_ok=True)

COLUMNS = {
    'frame': 'ndarray',
    'subtitle': 'ndarray'
}
COMPRESSION = 'zstd'
HASHES = 'sha1', 'xxh64'

with MDSWriter(out=SAVE_PATH, columns=COLUMNS, compression=COMPRESSION, hashes=HASHES) as save_file:
    for video_data in videos:
        save_file.write(video_data)
