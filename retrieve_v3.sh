#!/bin/bash
python3 bm25_retrieve_v3.py \
    --question_path ./CompetitionDataset/dataset/preliminary/questions_preliminary.json \
    --source_path ./CompetitionDataset/ref_contents_langchain \
    --output_path ./CompetitionDataset/dataset/preliminary/pred_retrieve_v3.json \
    --load_path ./custom_dicts/with_frequency
