from gensim.models import KeyedVectors
import os
# 載入預訓練的 Word2Vec 模型
current_dir = os.path.dirname(os.path.abspath(__file__))
pretrained_model_path = os.path.join(current_dir, 'wiki.zh.bin')
model = KeyedVectors.load(pretrained_model_path, mmap='r')
print('lodding model success')

# 使用模型
def evaluate_model(model, test_words=['年度', '財務', '股利', '報告']):
    print("\nModel Evaluation:")
    for word in test_words:
        if word in model.key_to_index:
            similar_words = model.most_similar(word, topn=5)
            print(f"\nMost similar to '{word}':")
            for similar_word, score in similar_words:
                print(f"  {similar_word}: {score:.4f}")
        else:
            print(f"\nWord '{word}' not in vocabulary")

evaluate_model(model)
