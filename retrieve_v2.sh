#!/bin/bash
python3 bm25_retrieve_v2.py \
    --question_path ./CompetitionDataset/dataset/preliminary/questions_preliminary.json \
    --source_path ./CompetitionDataset/reference \
    --output_path ./CompetitionDataset/dataset/preliminary/pred_retrieve.json \
    --load_path ./custom_dicts/with_frequency
