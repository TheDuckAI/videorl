import os
import json
import imageio  # NOTE: I tried to use cv2.VideoCapture, but it didn't work well.
import numpy as np
from streaming import MDSWriter
from tqdm import tqdm

BASE_DIR = "/root/projects/rl-nlp/videos/dataset_downsampled/00000/"  # NOTE: video2dataset creates more directories other than 00000. I used only 00000 because my machine doesn't have enough space.

def convert_time_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds


SAVE_PATH = "/root/projects/rl-nlp/videos/data_downsampled/mds-split/00000"
os.makedirs(SAVE_PATH, exist_ok=True)

COLUMNS = {
    'frame': 'ndarray',
    'subtitle': 'ndarray'
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

        # NOTE: To avoid a double loop of subtitle_data and reader for frames, store only the contents of subtitle_data in a numpy array and access the element later with np.where. 
        subtitle_texts = []
        start_frame_numbers = []
        end_frame_numbers = []

        with open(json_file_path, 'r') as f:
            data = json.load(f)
            fps = reader.get_meta_data()['fps']

            # fps = data["yt_meta_dict"]["info"]["fps"]  # NOTE: This is not always available
            try:
                subtitle_data_list = data["yt_meta_dict"]["subtitles"]  # NOTE: This is not always available
            except KeyError:
                continue

            for subtitle_data in subtitle_data_list:
                start_time = convert_time_to_seconds(subtitle_data["start"])  # Start time of the subtitle.
                end_time = convert_time_to_seconds(subtitle_data["end"])  # End time of the subtitle.

                start_frame_number = int(start_time * fps)
                end_frame_number = int(end_time * fps)

                subtitle_texts.append(np.frombuffer(subtitle_data["lines"][0].encode('utf-8'), dtype=np.uint8))
                start_frame_numbers.append(start_frame_number)
                end_frame_numbers.append(end_frame_number)

        try:
            MAX_LENGTH = max([len(sub) for sub in subtitle_texts])
        except ValueError:
            print('max() arg is an empty sequence')  # NOTE: If all intervals of subtitles are less than 1 second, we get an empty sequence.
            continue

        subtitle_texts = [np.pad(sub, (0, MAX_LENGTH - len(sub))) for sub in subtitle_texts]  # NOTE: Pad each subtitle to the same length

        subtitle_texts = np.asarray(subtitle_texts)
        start_frame_numbers = np.asarray(start_frame_numbers)
        end_frame_numbers = np.asarray(end_frame_numbers)

        frames = []
        subtitles = []
        missing_value_array = np.full(subtitle_texts[0].shape, 255, dtype=np.uint8)  # NOTE: This value is used to fill the missing subtitles.

        for frame_number, frame in enumerate(reader):

            #  Add subtitles to the list
            start_frame_pos_in_array = np.where(start_frame_numbers == frame_number)[0]  # NOTE: If there are multiple subtitles in the same frame, we use the first one.
            end_frame_pos_in_array = np.where(end_frame_numbers == frame_number)[-1]  # NOTE: If there are multiple subtitles in the same frame, we use the last one.

            if len(start_frame_pos_in_array) == 0 and len(end_frame_pos_in_array) == 0:
                subtitles.append(missing_value_array)  # NOTE If there is no subtitle in the frame, we use the missing value.
            elif len(start_frame_pos_in_array) == 0:
                subtitles.append(subtitle_texts[end_frame_pos_in_array[0]])  # NOTE: If the current frame number only matches an end frame number, a subtitle corresponding to that end frame is added.
            elif len(end_frame_pos_in_array) == 0:
                subtitles.append(subtitle_texts[start_frame_pos_in_array[0]])  # NOTE: If the current frame number only matches a start frame number, a subtitle corresponding to that start frame is added.
            else:
                subtitles.append(subtitle_texts[start_frame_pos_in_array[0]])  # NOTE: If the current frame number matches both a start frame number and an end frame number, a subtitle corresponding to that start frame is added. This can be that of an end frame.

            # Add frames to the list
            frame_array = np.array(frame)
            frames.append(frame_array)

        chunk_unit = 5
        num_chunks, remainder = divmod(len(frames), chunk_unit)

        if remainder != 0:
            chunked_frame = np.array_split(np.asarray(frames[:-remainder], dtype=np.float32), num_chunks, axis=0)
            chunked_subtitle = np.array_split(np.asarray(subtitles[:-remainder]), num_chunks, axis=0)

        video_data = {
            'frame': np.asarray(chunked_frame),
            'subtitle': np.asarray(chunked_subtitle)
        }

        save_file.write(video_data)
