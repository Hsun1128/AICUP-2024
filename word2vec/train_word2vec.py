import os
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import numpy as np

# 超參數設置
sg = 1  # Skip-gram model
vector_size = 300
window = 3
min_count = 3
workers = multiprocessing.cpu_count()
negative = 20
epochs = 60
initial_alpha = 0.015  # 更合理的初始學習率
min_alpha = 0.0001    # 最小學習率
alpha_decay = 5 * (initial_alpha - min_alpha) / epochs  # 學習率衰減

class ProgressCallback(CallbackAny2Vec):
    """Callback to show training progress and loss"""
    def __init__(self, epochs):
        self.epochs = epochs
        self.epoch = 0
        self.losses = []
        self.previous_loss = 0
        self.pbar = tqdm(total=epochs, desc="Training Progress")
    
    def on_epoch_begin(self, model):
        self.epoch += 1
        print(f"\nEpoch {self.epoch}/{self.epochs}")
        # 動態調整學習率
        model.alpha = initial_alpha - (self.epoch * alpha_decay)
        model.min_alpha = model.alpha
        print(f"Current learning rate: {model.alpha:.6f}")
    
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.previous_loss > 0:
            current_loss = loss - self.previous_loss
        else:
            current_loss = loss
        self.previous_loss = loss
        self.losses.append(current_loss)
        self.pbar.update(1)
        print(f"Epoch Loss: {current_loss:.4f}")
    
    def on_train_end(self, model):
        self.pbar.close()

# 設定文件路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
corpus_file = os.path.join(current_dir, 'corpusSegDone.txt')
pretrained_model_path = os.path.join(current_dir, 'wiki.zh.bin')
save_file = os.path.join(current_dir, 'model.bin')

# 載入預訓練向量
print("Loading pretrained vectors...")
pretrained_vectors = KeyedVectors.load(pretrained_model_path, mmap='r')

# 初始化模型
model = Word2Vec(
    sg=sg,
    vector_size=vector_size,
    window=window,
    min_count=min_count,
    workers=workers,
    alpha=initial_alpha,
    min_alpha=min_alpha,
    negative=negative,
)

# 建立詞彙表並初始化向量
print("Building vocabulary...")
model.build_vocab(LineSentence(corpus_file))

# 初始化預訓練向量
print("Initializing vectors from pretrained model...")
pretrained_keys = list(set(pretrained_vectors.key_to_index.keys()) & set(model.wv.key_to_index.keys()))
for word in pretrained_keys:
    model.wv.vectors[model.wv.key_to_index[word]] = pretrained_vectors[word]

# 訓練模型
print("Training model...")
model.train(
    corpus_iterable=LineSentence(corpus_file),
    total_examples=model.corpus_count,
    epochs=epochs,
    compute_loss=True,
    callbacks=[ProgressCallback(epochs=epochs)]
)

# 保存模型
print("Saving model...")
model.save(save_file)

# 簡單的模型評估
def evaluate_model(model, test_words=['年度', '財務', '股利', '報告']):
    print("\nModel Evaluation:")
    for word in test_words:
        if word in model.wv:
            similar_words = model.wv.most_similar(word, topn=5)
            print(f"\nMost similar to '{word}':")
            for similar_word, score in similar_words:
                print(f"  {similar_word}: {score:.4f}")
        else:
            print(f"\nWord '{word}' not in vocabulary")

evaluate_model(model)
