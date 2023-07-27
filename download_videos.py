from video2dataset import video2dataset

if __name__ == '__main__':
    video2dataset(
        url_list='hd_vila_test.parquet',
        input_format='parquet',
        output_format='webdataset',
        output_folder='hd_vila_test_new2',
        url_col="url",
        enable_wandb=False,
        encode_formats={"video": "mp4", "audio": "m4a"},
        config="config.yaml"
    )