import csv
import yaml
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

# Mocking the v2d data fetch and image download for the sake of demonstration
from mock_utils import fetch_v2d_data, download_images

def compute_similarity(img1, img2, metric):
    """
    Compute the similarity between two images based on the specified metric.
    
    Args:
    - img1, img2: Images to be compared.
    - metric (str): Metric to be used (e.g., "SSIM", "MSE").
    
    Returns:
    - float: Similarity score.
    """
    if metric == "SSIM":
        return ssim(img1, img2, multichannel=True)
    elif metric == "MSE":
        return mse(img1, img2)
    # Add other metrics as needed
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def compute_quality_score(video_id, config):
    """
    Compute the quality score for a given video based on its ID and provided configuration.
    
    Args:
    - video_id (str): ID of the video.
    - config (dict): Configuration parameters including weights and metric choice.
    
    Returns:
    - float: Computed quality score.
    """
    # Mock download of images
    images = download_images(video_id)

    # Compute pairwise differences
    diffs = []
    for i in range(3):
        for j in range(i+1, 4):
            diffs.append(compute_similarity(images[i], images[j], config["metric"]))

    movement_score = sum(diffs) / len(diffs)

    # Mock fetching v2d data
    v2d_data = fetch_v2d_data(video_id)
    resolution = int(v2d_data["yt_meta_dict"]["info"]["resolution"].split('x')[1])

    # Compute quality score
    quality_score = config["w1"] * movement_score + config["w2"] * resolution

    return quality_score

def split_data(input_csv_path, train_csv_path, test_csv_path, test_size=0.2):
    """
    Split the provided data into train and test datasets.
    
    Args:
    - input_csv_path (str): Path to the input CSV file.
    - train_csv_path (str): Path to save the train split.
    - test_csv_path (str): Path to save the test split.
    - test_size (float): Proportion of data to include in the test split.
    """
    
    # Read the data from the input CSV
    with open(input_csv_path, mode='r') as infile:
        reader = list(csv.DictReader(infile))
        
        # Split the data
        train_data, test_data = train_test_split(reader, test_size=test_size, random_state=42)
        
        # Write the train split to the train CSV
        with open(train_csv_path, mode='w', newline='') as trainfile:
            writer = csv.DictWriter(trainfile, fieldnames=reader[0].keys())
            writer.writeheader()
            for row in train_data:
                writer.writerow(row)
                
        # Write the test split to the test CSV
        with open(test_csv_path, mode='w', newline='') as testfile:
            writer = csv.DictWriter(testfile, fieldnames=reader[0].keys())
            writer.writeheader()
            for row in test_data:
                writer.writerow(row)

def main(input_csv_path, output_csv_path, config_path):
    """
    Main function to compute quality scores for all videos in the input CSV and save to the output CSV.
    
    Args:
    - input_csv_path (str): Path to the input CSV file containing video data.
    - output_csv_path (str): Path to the output CSV file to save results.
    - config_path (str): Path to the config.yaml file.
    """
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    with open(input_csv_path, mode='r') as infile, open(output_csv_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["quality_score"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            video_id = row["id"]
            quality_score = compute_quality_score(video_id, config)
            row["quality_score"] = quality_score
            writer.writerow(row)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process and compute quality scores for videos.')
    
    parser.add_argument('input_csv', help='Path to the input CSV file containing video data.')
    parser.add_argument('train_csv', help='Path to save the train split of the data.')
    parser.add_argument('test_csv', help='Path to save the test split of the data.')
    parser.add_argument('config', help='Path to the config.yaml file.')
    
    args = parser.parse_args()
    
    # Read test size from config
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    test_size = config.get("split_test_size", 0.2)  # Default test size is 0.2 if not specified in config
    
    # Split the data into train and test splits
    split_data(args.input_csv, args.train_csv, args.test_csv, test_size)
