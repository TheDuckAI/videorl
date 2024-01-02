"""
Generate a sample CSV file of size r from the original CSV file.
Generate train and test CSV files from the original CSV file.
"""
import csv
import os 
import argparse
from sklearn.model_selection import train_test_split

def extract_first_r_entries(input_csv, output_csv, r):
    """
    Extracts the first r entries from the input CSV and writes them to the output CSV.
    
    Args:
    - input_csv (str): Path to the input CSV file.
    - output_csv (str): Path to the output CSV file.
    - r (int): Number of entries to extract.
    """
    with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header
        writer.writerow(next(reader))
        
        # Write first r entries
        for _ in range(r):
            try:
                writer.writerow(next(reader))
            except StopIteration:
                break

def split_data(input_csv_path, train_csv_path, test_csv_path, test_size=0.2):
    """
    Split the provided data into train and test datasets.
    """
    with open(input_csv_path, mode='r') as infile:
        reader = list(csv.DictReader(infile))
        train_data, test_data = train_test_split(reader, test_size=test_size, random_state=42)

        with open(train_csv_path, mode='w', newline='') as trainfile:
            writer = csv.DictWriter(trainfile, fieldnames=reader[0].keys())
            writer.writeheader()
            for row in train_data:
                writer.writerow(row)

        with open(test_csv_path, mode='w', newline='') as testfile:
            writer = csv.DictWriter(testfile, fieldnames=reader[0].keys())
            writer.writeheader()
            for row in test_data:
                writer.writerow(row)



if __name__ == "__main__":
    r = 100

    input_csv="sample/videos.csv" # original file 
    output_csv = "cache/sample.csv" 
    
    if os.path.exists(output_csv):
        print(f"File {output_csv} already exists. Skipping.")
    else:
        extract_first_r_entries(input_csv, output_csv, r)


    import argparse
    parser = argparse.ArgumentParser(description='Compute quality scores for videos.')
    parser.add_argument('input_csv', help='Path to the input CSV file containing video data.')
    parser.add_argument('train_csv', help='Path to the train CSV file.')
    parser.add_argument('test_csv', help='Path to the test CSV file.')
    args = parser.parse_args()

    split_data(args.input_csv, args.train_csv, args.test_csv)