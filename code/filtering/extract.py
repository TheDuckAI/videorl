import csv
import argparse

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

if __name__ == "__main__":
    r = 100
    input_csv="../../sample/videos.csv"
    output_csv = "cache/sample.csv"
    
    extract_first_r_entries(input_csv, output_csv, r)
