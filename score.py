import json
import os
import logging

logging.basicConfig(level=logging.INFO, filename='score.log', filemode='a', format='%(asctime)s:%(message)s')
logger = logging.getLogger(__name__)

def calculate_score(ground_truth_file, prediction_file):
    # Load the ground truth and prediction data
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)["ground_truths"]
    
    with open(prediction_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)["answers"]
    
    ground_truth_dict = {item["qid"]: item["retrieve"] for item in ground_truths}
    
    score = 0
    wrong_predictions = []
    
    # Calculate score and track wrong predictions
    for prediction in predictions:
        qid = prediction["qid"]
        predicted_retrieve = prediction["retrieve"]
        
        if qid in ground_truth_dict:
            if ground_truth_dict[qid] == predicted_retrieve:
                score += 1
            else:
                wrong_predictions.append({
                    'qid': qid,
                    'predicted': predicted_retrieve,
                    'correct': ground_truth_dict[qid]
                })
    
    total_questions = len(predictions)
    average_score = score / total_questions if total_questions > 0 else 0
    
    return score, average_score, wrong_predictions

def main(ground_truth_file_path, prediction_file_path):
    # Calculate and logger.info the score
    score, average_score, wrong_predictions = calculate_score(ground_truth_file_path, prediction_file_path)
    logger.info(f"Total Score: {score}")
    logger.info(f"Average Score: {average_score:.2f}")
    logger.info("\n錯誤的預測: ")
    for wrong in wrong_predictions:
        logger.info(f"QID: {wrong['qid']}")
        logger.info(f"預測值: {wrong['predicted']}")
        logger.info(f"正確答案: {wrong['correct']}")
        logger.info("-" * 50)
    logger.info('end\n')
    print(f"Total Score: {score}")
    print(f"Average Score: {average_score:.2f}")

if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create paths relative to the script location
    ground_truth_file = os.path.join(current_dir, '競賽資料集/dataset/preliminary/ground_truths_example.json')
    prediction_file = os.path.join(current_dir, '競賽資料集/dataset/preliminary/pred_retrieve.json')

    main(ground_truth_file, prediction_file)
