# video-dataset
## Quick Start
### Download HD-VILA-100M Dataset
- https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m
### Extract URLs and Save as CSV
```bash
cd preprocess
python extract_url.py
```
### Get Videos and Metadata with video2dataset
```bash
video2dataset \\
--url_list="results/all_urls.csv"  \\
--url_col="url" \\
--output_folder="dataset"
```
- https://github.com/iejMac/video2dataset
### Split Videos & Subtitles and Save them as StreamingDataset
```bash
cd preprocess
python split_clips.py
```