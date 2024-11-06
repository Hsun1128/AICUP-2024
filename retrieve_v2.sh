#!/bin/bash
python3 bm25_retrieve_v2.py \
    --question_path ./競賽資料集/dataset/preliminary/questions_example.json \
    --source_path ./競賽資料集/reference \
    --output_path ./競賽資料集/dataset/preliminary/pred_retrieve.json \
    --load_path ./custom_dicts/with_frequency
