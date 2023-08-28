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

# videos = []

SAVE_PATH = "/root/projects/rl-nlp/videos/data/mds-not-split/00000"
os.makedirs(SAVE_PATH, exist_ok=True)

COLUMNS = {
    'frame': 'ndarray',
    'subtitle': 'ndarray'
}
COMPRESSION = 'zstd'
HASHES = 'sha1', 'xxh64'

with MDSWriter(out=SAVE_PATH, columns=COLUMNS, compression=COMPRESSION, hashes=HASHES) as save_file:        

    for video_filename in tqdm(os.listdir(BASE_DIR)[:10]):  # NOTE: I used only 10 videos for the test because the space of my machine is limited.
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

            for i, frame in enumerate(reader):

                if frame is None:
                    break

                current_time = i / fps

                frame_array = np.array(frame)
                frames.append(frame_array)

                subtitle_value = np.array([-1])  # NOTE: -1 means no subtitle. if the current time is in the interval of the subtitle, we overwrite this value.

                for subtitle in subtitle_data:
                    start_time = convert_time_to_seconds(subtitle["start"])  # Start time of the subtitle.
                    end_time = convert_time_to_seconds(subtitle["end"])  # End time of the subtitle.

                    if start_time <= current_time and current_time < end_time:  # NOTE: If the current time is in the interval of the subtitle, we save the subtitle. 
                        subtitle_value = np.frombuffer(subtitle["lines"][0].encode('utf-8'), dtype=np.uint8)
                        break

                subtitles.append(subtitle_value)            

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

        save_file.write(video_data)
    # videos.append(video_data)
