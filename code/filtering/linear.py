import wandb
import csv
import yaml
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils import compute_quality_score
import joblib

# Initialize wandb
wandb.init(project="video_quality_prediction")

def load_data(input_csv_path, config_path):
    """
    Load data from the provided CSV file and compute quality scores if needed.
    
    Args:
    - input_csv_path (str): Path to the input CSV file containing video data.
    
    Returns:
    - X (np.array): Features for each video.
    - y (np.array): Quality scores for each video.
    """
    data = []
    with open(input_csv_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        data = [row for row in reader]

    # Check if "quality_score" column exists
    if "quality_score" not in data[0]:
        # Compute quality scores and add them to the data
        for row in data:
            video_id = row["id"]
            quality_score = compute_quality_score(video_id, config_path)  # This function is from utils.py
            row["quality_score"] = quality_score

        # Write the updated data back to the CSV file
        with open(input_csv_path, mode='w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=data[0].keys())
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    # Extract features and quality scores for training
    features = []
    quality_scores = []
    for row in data:
        video_id = row["id"]
        quality_score = float(row["quality_score"])
        feature_vector = [
            float(row["length"].split(":")[0]),  # Taking video length in minutes as a feature
        ]
        features.append(feature_vector)
        quality_scores.append(quality_score)

    return np.array(features), np.array(quality_scores)


def train_model(input_csv_path, model_path, config_path):
    """
    Train a Linear Regression model on the provided data.
    
    Args:
    - input_csv_path (str): Path to the input CSV file containing video data.
    - model_path (str): Path to save the trained model.
    - config_path (str): Path to the config.yaml file.
    """
    
    with open(config_path, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
    X, y = load_data(input_csv_path, config_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["split_test_size"], random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Compute MSE
    mse_val = mean_squared_error(y_test, y_pred)
    
    # Log metrics to wandb
    wandb.log({"MSE": mse_val})
    
    # Save the model locally
    joblib.dump(model, model_path)
    
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a Linear Regression model for video quality prediction.')
    parser.add_argument('input_csv', help='Path to the input CSV file containing video data.')
    parser.add_argument('model_path', help='Path to save the trained model.')
    parser.add_argument('config', help='Path to the config.yaml file.')
    args = parser.parse_args()
    
    train_model(args.input_csv, args.model_path, args.config)
