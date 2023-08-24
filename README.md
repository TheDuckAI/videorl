# Data Pipeline using Video2Dataset
## [HDVILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m)
HDVILA 100M is a dataset of 100M high-resolution videos from YouTube.

### Download the metdata
First, run `wget -O hdvila100m.zip https://hdvila.blob.core.windows.net/dataset/hdvila100m.zip?sp=r&st=2022-06-28T03:33:11Z&se=2026-01-01T11:33:11Z&spr=https&sv=2021-06-08&sr=b&sig=VaqQkLFDqKinfkaPNs1jJ1EQIYCB%2FUPYiqFqmjWye6Y%3D` to download the HD VILA 100M metadata. Next, just run `unzip hdvilla100m.zip` in order to unzip the metadata. You should now have an `hdvila100m/` directory.

Next, we need to do some preprocessing to get this metadata formatted into a nice parquet. The create_parquet.py script will take the downloaded metadata `.jsonl` files and create a parquet with all the relevant information.

Once you run this, you should have a file `hd_vila.parquet` with all the relevant metadata.

### Create the config file for video processing

In the config.yaml file, we will define how we want to process the videos. In the example 'config.yaml' file, we decrease the number of samples per shard (a good heuristic is having shards be ~1Gb), detect cuts for each video and place them in the metadata (using CutDetectionSubSampler), and auto generate subtitles based on the audio for each video (using WhisperSubSampler).

Further explanation on config.yaml file and details on subsampler can be found in video2dataset/API.md.

### Downloading + Cut Detection + Subtitle Generation

Run the command 'sbatch hd_vila_test.sbatch' in JUWELS to start running the download job. The hd_vila_test.sbatch essentially runs 'download_videos.py', which runs video2dataset on the input parquet and saves the videos, audio and subtitles along with the metadata in the webdataset format.

An example of the processed data can be found in:
https://drive.google.com/file/d/1qOblEj6xXb0nEofJeRxO7xv91wmYlH7w/view?usp=sharing
