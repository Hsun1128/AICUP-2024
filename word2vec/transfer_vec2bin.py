from gensim.models import KeyedVectors
import os
# 載入預訓練的 Word2Vec 模型
# 假設你已經下載了 Google 的 Word2Vec 模型文件 word2vec.bin
current_dir = os.path.dirname(os.path.abspath(__file__))
pretrained_model_path = os.path.join(current_dir, 'wiki.zh.vec')
save_model_path = os.path.join(current_dir, 'wiki.zh.bin')
model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=False)
print('lodding model success')

# 保存為 .bin 格式
model.save(save_model_path)
print('transfer model success')

