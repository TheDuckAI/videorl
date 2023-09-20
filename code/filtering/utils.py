import csv
import yaml
import os
import json

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from video2dataset import video2dataset

def compute_similarity(img1, img2, metric):
    """
    Compute the similarity between two images based on the specified metric.
    """
    if metric == "SSIM":
        return ssim(img1, img2, multichannel=True)
    elif metric == "MSE":
        return mse(img1, img2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def compute_quality_score(video_file_path, config):
    """
    Compute the quality score for a given video file based on the provided configuration.
    """
    # For demonstration purposes, assuming the video_file_path can directly be used
    # to extract relevant frames. In a real-world scenario, a video processing library would be used.
    
    # Compute pairwise differences
    diffs = []
    for i in range(3):
        for j in range(i+1, 4):
            diffs.append(compute_similarity(video_file_path, video_file_path, config["metric"])) # Placeholder

    movement_score = sum(diffs) / len(diffs)

    # For now, using a placeholder for resolution. In a real-world scenario, the resolution would be fetched.
    resolution = 1080

    # Compute quality score
    quality_score = config["w1"] * movement_score + config["w2"] * resolution

    return quality_score


def main(input_csv_path, output_csv_path, config_path):
    """
    Main function to compute quality scores for all videos and save to the output CSV.
    """
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Use video2dataset to download videos and save them to a designated folder
    video2dataset(url_list=input_csv_path,
                  output_folder="dataset",
                  url_col=config["url_col"],
                  caption_col=config["caption_col"])

    with open(input_csv_path, mode='r') as infile, open(output_csv_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["quality_score"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            video_id = row["id"]
            
            # Assuming video files are saved with format {video_id}.mp4 and metadata in {video_id}.json
            video_file_path = os.path.join("dataset", f"{video_id}.mp4")
            metadata_file_path = os.path.join("dataset", f"{video_id}.json")
            
            with open(metadata_file_path, 'r') as json_file:
                metadata = json.load(json_file)
            
            # Use metadata if necessary for further processing
            
            quality_score = compute_quality_score(video_file_path, config)
            row["quality_score"] = quality_score
            writer.writerow(row)

