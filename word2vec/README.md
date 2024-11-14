# 📚 Word2Vec Training

This directory provides tools for building a Word2Vec training dataset using specific text sources, such as insurance, finance, and FAQ data.

## 🔄 Data Preprocess

To begin, configure the following parameters in the [`segment_corpus.py`](segment_corpus.py) script:

```python
    source_path_insurance = os.path.join(current_dir, '../競賽資料集/reference/insurance')  # Set the insurance data path
    source_path_finance = os.path.join(current_dir, '../競賽資料集/reference/finance')  # Set the finance data path
    json_data_path = os.path.join(current_dir, '../競賽資料集/reference/faq/pid_map_content.json')  # Set the JSON data path
    stopwords_filepath = os.path.join(current_dir, '../custom_dicts/stopwords.txt')  # Set the stop words file path
    output_filepath = os.path.join(current_dir, 'corpusSegDone.txt')  # Set the output file path
    load_path = os.path.join(current_dir, '../custom_dicts/with_frequency')  # Set the custom dictionary path
    zhTW_dict_path = os.path.join(current_dir, '../custom_dicts/dict.txt.big')  # Set the traditional Chinese dictionary path
```

These parameters allow you to define the locations of the source text files, custom dictionaries, stopwords, and the output path for the processed corpus.

## ⚡️ Usage
After setting the correct paths, run the following script to process the text and generate the training corpus:
```bash
python3 segment_corpus.py
```

The processed data will be saved automatically as [`corpusSegDone.txt`](corpusSegDone.txt).

> [!TIP]
> Additional information and logs regarding the processing can be found in [`segment_corpus.log`](segment_corpus.log).

For further information on customizing the stopwords or dictionaries, refer to the corresponding sections in the [`custom_dicts/README.md`](../custom_dicts/README.md).

## 📦 Download Word2Vec Pre-trained Model

To download the pre-trained Word2Vec model and convert it to a faster binary format, follow these steps:

### 1. **Download and Convert the Model**

Run the following script to automatically download the pre-trained Word2Vec model named `wiki.zh.vec` and convert it to the `.bin` format, which is optimized for faster loading:

```bash
python3 transfer_vec2bin.py
```

This will download the model and save it as `wiki.zh.vec` in the current directory, then convert it into the .bin format and store it as `wiki.zh.bin`.

### 2. **Verify the Conversion**
After the conversion, you can use the load_pretrain.py script to verify that the wiki.zh.bin model is working properly:

```bash
python3 load_pretrain.py
```

This script checks if the wiki.zh.bin model can be loaded successfully and confirms that it is ready for use.

### 🚀 Benefits
Faster Loading: The .bin format is optimized for quicker model loading and inference, making it ideal for performance-critical applications.
Pre-trained Model: The wiki.zh.vec model is trained on large Chinese text corpora, providing a solid foundation for various natural language processing (NLP) tasks.

### 🌐 Download Pre-trained Models
You can also find other pre-trained models that suit your needs. For example, the Word2Vec pre-trained vectors are available for download at the following link:

- [Word2Vec Pre-trained Vectors](https://fasttext.cc/docs/en/pretrained-vectors.html)

## 🧠 Train Your Own Word2Vec Model

Before training the model, you need to set the hyperparameters in [`train_word2vec`](train_word2vec). Here is an example configuration:

```python
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
```
**Explanation of Key Hyperparameters:**
- `sg`: Determines the training algorithm. Set to 1 for Skip-gram (recommended for smaller datasets) and 0 for CBOW (Continuous Bag of Words).
- `vector_size`: The size of the word vectors to be created (typically 100–500 dimensions).
- `window`: The size of the context window used for training.
- `min_count`: Words with total frequency lower than this value will be ignored.
- `workers`: The number of CPU cores to use for parallel processing.
- `negative`: The number of negative samples for each positive word pair.
- `epochs`: The number of iterations over the corpus.
- `initial_alpha`: The initial learning rate for the model.
- `min_alpha`: The minimum learning rate that the model will decay to.
- `alpha_decay`: The rate at which the learning rate will decay.

### 🚀 Start Training
After setting up the hyperparameters, you can begin training your Word2Vec model using the following command:

```bash
python3 train_word2vec.py
```

Once the training is completed, congratulations! You will have your own Word2Vec model ready for use in your natural language processing tasks.

## 🏆 Outcome
After running the training process, you'll have a custom-trained Word2Vec model that you can use for:

Word embeddings  
Semantic analysis  
Text similarity  
And much more!  
Enjoy your trained model!  
