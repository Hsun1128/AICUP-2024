import json
import os
from tqdm import tqdm
import argparse

def main(folder, input_root, output_root):
    if os.path.exists(os.path.join(output_root, folder)) is False:
        os.makedirs(os.path.join(output_root, folder))

    with open(os.path.join(input_root, folder, 'pid_map_content.json'), 'r') as f:
        data = json.load(f)
        
    files = list(data.keys())

    for file in tqdm(files):
        text = ""
        questions = data[file]
        for question in questions:
            text += question['question'] + '\t'
            text += ", ".join(question['answers']) + '\n'

        output = {
            "id":file,
            "contents":text
        }

        # save pdf_text to json
        with open(os.path.join(output_root, folder, f'{output["id"]}.json'), 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

# python faq_format.py --folder faq --input_root ../CompetitionDataset/reference --output_root ../CompetitionDataset/ref_contents
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)
    parser.add_argument('--input_root', type=str)
    parser.add_argument('--output_root', type=str)
    args = parser.parse_args()

    main(args.folder, args.input_root, args.output_root)