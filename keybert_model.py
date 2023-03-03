import pickle
import config
import os
import json
import stanza
from sklearn.feature_extraction.text import CountVectorizer

# preprocessing custom data
def preprocessing_keybert_data(name_dataset):
    """Processes the document as a list of sentences and saves it as txt file.

    Args:
        name_dataset (str): Name of the dataset
    """

    with open(os.path.join(config.ENV_PATH, config.DATA_PATH, name_dataset + '.json')) as f:
        json_data = json.load(f)

    keybert_docs, keybert_keywords, keybert_summary = [], [], []

    for l in json_data:
        keybert_docs.append(l['dialogue'])
        # keybert_keywords.append(l['finegrained_relevant_dialogue'])
        # keybert_summary.append(l['summary'])

    keybert_docs_path = os.path.join(config.ENV_PATH, 'KeyBERT/', config.DATA_PATH, f"keybert_docs_{name_dataset}.txt")
    # keybert_keywords_path = os.path.join(config.ENV_PATH, 'KeyBERT/', config.DATA_PATH, f"keybert_keywords_{name_dataset}.txt")
    # keybert_summary_path = os.path.join(config.ENV_PATH, 'KeyBERT/', config.DATA_PATH, f"keybert_summary_{name_dataset}.txt")
    with open(keybert_docs_path, 'wb') as outfile_docs:
        pickle.dump(keybert_docs, outfile_docs)
    # with open(keybert_keywords_path, 'w') as outfile_keywords:
    #     pickle.dump(keybert_keywords, outfile_keywords)
    # with open(keybert_summary_path, 'w') as outfile_summary:
    #     pickle.dump(keybert_summary, outfile_summary)

def map_keyword2extractedsentence(docs, keywords, k):
    """Given some keywords, the relevant sentences are those sentences containing at least k tokens.
    TODO: does not yet take into account the n-grams, thus overestimates the number of relevant sentences.

    Args:
        train_docs (list): List of train documents.
        keywords (list of list): List of list of keywords.
        k (int): Threshold for the number of keywords such that a sentence is relevant.

    Returns:
        summary (list): List of relevant sentences for each dialogue.
    """
    nlp = stanza.Pipeline(lang='en', processors='tokenize')

    summary =[[] for _ in range(len(docs))]
    # takes every dialogue of the train_docs
    for i, dialogue in enumerate(docs):
        # get all unique keyword tokens returned by keybert for each sentence
        # unique_keywords_dialoge = set()
        # for keywords_dialoge in keywords[i]:
        #     deconstruct the n-gram keyphrases into tokens
        #     for keyphrase in keywords_dialoge[0]:
        #         try:
        #             keyphrase_tokenized = nlp(keyphrase)  # [0]  # TODO: it should be with n-grams and not with tokens
        #             for key_token in keyphrase_tokenized.iter_tokens():
        #                 unique_keywords_dialoge.add(key_token.text)
        #         except AssertionError as e:
        #             print(f"The keyphrase {keyphrase} could not be parsed, with error: {e}")
        #             continue

        # takes every sentence of the dialogue
        dialogue_tokenized = nlp(dialogue)
        for j, sentence in enumerate(dialogue_tokenized.sentences):
            count_keywords = 0

            # normalize the sentence, as keyBERT did
            sentence_normal = sentence.text.lower()
            stop_words = CountVectorizer(stop_words='english').get_stop_words()
            sentence_tokenized = nlp(sentence_normal)
            for token in sentence_tokenized.iter_tokens():
                if token.text in stop_words:
                    sentence_normal = sentence_normal.replace(token.text, '')

            for keyword_ngram_dialoge in keywords[i]:
                if keyword_ngram_dialoge[0] in sentence_normal:
                    count_keywords += 1
            if count_keywords >= k:
                summary[i].append(sentence.text)

            # # if the token is a keyword, add the sentence to the summary
            # for token in sentence.tokens:
            #         for keyword_token in unique_keywords_dialoge:
            #             if token.text == keyword_token:
            #                 count_keywords += 1
            # if count_keywords >= k:
            #     # get the whole sentence and append it to the summary
            #     reconstructed_sentence = ""
            #     for token in sentence.tokens:
            #         reconstructed_sentence += token.text + " "
            #     summary[i].append(reconstructed_sentence)
    return summary