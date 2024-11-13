#!/bin/bash
# Check if word2vec/wiki.zh.bin exists
if [ ! -f "word2vec/wiki.zh.bin" ]; then
    echo "Converting word2vec model to binary format..."
    python3 word2vec/transfer_vec2bin.py
fi

python3 bm25_retrieve_v2.py \
    --question_path ./CompetitionDataset/dataset/preliminary/questions_example.json \
    --source_path ./CompetitionDataset/reference \
    --output_path ./CompetitionDataset/dataset/preliminary/pred_retrieve.json \
    --load_path ./custom_dicts/with_frequency
