#!/bin/bash

# 入力ディレクトリと出力ディレクトリの設定
INPUT_DIR="/root/projects/rl-nlp/videos/dataset/00000"
OUTPUT_DIR="/root/projects/rl-nlp/videos/dataset_downsampled/00000"

# 出力ディレクトリが存在しない場合、作成
mkdir -p "$OUTPUT_DIR"

# 入力ディレクトリ内の各.jsonファイルに対してループ処理
for input_file in "$INPUT_DIR"/*.json; do
  # ファイル名（拡張子あり）の取得
  filename=$(basename -- "$input_file")

  # 出力ファイルのパス設定
  output_file="$OUTPUT_DIR/$filename"

  # コピー操作
  cp "$input_file" "$output_file"
done
