from gensim.models import KeyedVectors
import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########################################################
# 下載預訓練的 Word2Vec 模型
########################################################
url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh.vec'
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'wiki.zh.vec')
if not os.path.exists(model_path):
    logger.info('Word2Vec model not found')
    logger.info('Downloading Word2Vec model...')
    urllib.request.urlretrieve(url, model_path)

########################################################
# 載入預訓練的 Word2Vec 模型
########################################################
logger.info('Loading Word2Vec model...')
current_dir = os.path.dirname(os.path.abspath(__file__))
pretrained_model_path = os.path.join(current_dir, 'wiki.zh.vec')
save_model_path = os.path.join(current_dir, 'wiki.zh.bin')
model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=False)
logger.info('lodding model success')

########################################################
# 保存為 .bin 格式
########################################################
model.save(save_model_path)
logger.info('transfer model success')

