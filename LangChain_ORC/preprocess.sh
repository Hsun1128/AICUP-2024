#!/bin/bash
echo "Preprocessing FAQ data"
python faq_format.py --folder faq --input_root ../CompetitionDataset/reference --output_root ../CompetitionDataset/ref_contents

echo "Preprocessing Finance data"
python pdf_extractor.py ../CompetitionDataset/reference/finance ../CompetitionDataset/ref_contents/finance

echo "Preprocessing Insurance data"
python pdf_extractor.py ../CompetitionDataset/reference/insurance ../CompetitionDataset/ref_contents/insurance
