#!/bin/bash

# 入力ディレクトリと出力ディレクトリの設定
INPUT_DIR="/root/projects/rl-nlp/videos/dataset/00000"
OUTPUT_DIR="/root/projects/rl-nlp/videos/dataset_downsampled/00000"

# 出力ディレクトリが存在しない場合、作成
mkdir -p "$OUTPUT_DIR"

# 入力ディレクトリ内の各.mp4ファイルに対してループ処理
for input_file in "$INPUT_DIR"/*.mp4; do
  # ファイル名（拡張子なし）の取得
  filename=$(basename -- "$input_file")
  filename_noext="${filename%.*}"

  # 出力ファイルのパス設定
  output_file="$OUTPUT_DIR/$filename_noext.mp4"

  # FFmpegを使ってfpsを1fpsに変換
  ffmpeg -i "$input_file" -vf "fps=1" "$output_file"
done
