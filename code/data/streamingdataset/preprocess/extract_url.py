import pandas as pd
import json
import os

dir_path = '../data/hdvila'
file_pattern = 'hdvila_part{}.jsonl'
output_csv = '../results/all_urls.csv'

all_urls = []

for i in range(11):  # hdvila_part0.jsonl to hdvila_part10.jsonl
    file_path = os.path.join(dir_path, file_pattern.format(i))
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            if 'url' in data:
                all_urls.append(data['url'])

df = pd.DataFrame(all_urls, columns=['url'])

df.to_csv(output_csv, index=False)
