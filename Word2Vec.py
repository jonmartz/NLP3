import spacy
import pandas as pd
import preprocessing as pre
import json
import gzip

"""
Save the Spacy word embeddings to disk.
"""

print('loading model...')
language_model = spacy.load('en_core_web_sm')  # dim=96
# language_model = spacy.load('en_core_web_md')  # dim=300

print('processing train and test sets...')
vocabulary, vocabulary_ordered, vectors = set(), [], []
# df_train = pd.read_csv('dataset_train.csv')
df_train = pre.create_tweets_df('trump_train.tsv')
# df_test = pd.read_csv('dataset_test.csv')
df_test = pre.create_tweets_df('trump_test.tsv')
texts = pd.concat([df_train['tweet text'], df_test['tweet text']])
for i, text in enumerate(texts):
    print('%d/%d' % (i + 1, len(texts)))
    words = pre.tokenize_for_vectors(text)
    for word in words:
        if word not in vocabulary:
            token = language_model(word)
            if token.vector_norm:
                vocabulary.add(word)
                vocabulary_ordered.append(word)
                vectors.append(token.vector.tolist())

print('saving vectors...')
for data, file_name in zip([vocabulary_ordered, vectors], ['vocab', 'vectors']):
    json_str = json.dumps(data) + '\n'
    with gzip.open('%s.sav' % file_name, 'w') as file:
        file.write(json_str.encode('utf-8'))
# pd.DataFrame({'word': vocabulary_ordered, 'vector': vectors}).to_csv('word_vecs.csv')
