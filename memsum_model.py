#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""All the functions for the memsum model."""

import os
import json
from tqdm import tqdm
import subprocess
import numpy as np

import stanza
import torch

import config

def prepare_memsum():
    """Get some data and prepare the folder structure.
    """
    os.chdir(config.ENV_PATH)

    # download the pretrained word embeddings
    if not os.path.exists(os.path.join(config.ENV_PATH, 'MemSum-015ddda/model/glove')):
        subprocess.run(["mkdir", "-p", "MemSum-015ddda/model", "&&", 
                        "cd",  "MemSum-015ddda/model", "&&", 
                        "gdown", "https://drive.google.com/drive/folders/1lrwYrrM3h0-9fwWCOmpRkydvmF6hmvmW", "--folder"])
    else:
        print("The pretrained word embeddings are already downloaded, not retrieving glove.")

    # get and run Stanford CoreNLP
    subprocess.run(['apt-get', 'install', 'openjdk-8-jdk-headless', '-qq', '>', '/dev/null'])

    # new files only if not already downloaded
    subprocess.run(['wget', '-nc', 'https://nlp.stanford.edu/software/stanford-corenlp-latest.zip'])
    subprocess.run(['unzip', '-n', 'stanford-corenlp-latest.zip'])

def preprocessing_memsum_data(name_dataset, tokens=False):
    """Takes the data from the json file and creates a jsonl file for the memsum model.

    Args:
        name_dataset (string): The name of the dataset. It is used to find the json file and to name the jsonl file.
        tokens (bool, optional): If True, the text is tokenized. Defaults to False.
    """
    # start the tokenizer
    nlp = stanza.Pipeline(lang='en', processors='tokenize')

    with open(os.path.join(config.ENV_PATH, config.DATA_PATH, name_dataset + '.json')) as f:
        json_data = json.load(f)

    memsum_training = os.path.join(config.ENV_PATH, 'MemSum-015ddda/', config.DATA_PATH, f"memsum_{'tokens' if tokens else 'sentences'}_{name_dataset}.jsonl")
    with open(memsum_training, 'w') as outfile:
        for l in json_data:
            dialog_temp = {}

            # the dialogue is the text
            text_temp = []
            sentences = l['dialogue'].split('#')
            for s in sentences:
                if tokens:
                    # tokenize the sentence
                    doc = nlp(s)
                    text_temp.extend([token.text for sent in doc.sentences for token in sent.tokens])
                else:
                    # remove empty strings
                    # to normalize between train and test, we need to remove the empty end of the sentence
                    s = s.strip('\n') # TODO: this accidently also strips stuff like ')'
                    if s == '':
                        continue
                    else:
                        if s[-1] == ' ': # TODO: this does not work as expected
                            text_temp.append(s[:-1])
                        else:
                            text_temp.append(s)
            dialog_temp['text'] = text_temp

            if tokens:
                # tokenize the summary
                summary_temp = []
                for s in l['finegrained_relevant_dialogue']:
                    # tokenize the sentence
                    doc = nlp(s)
                    summary_temp.extend([token.text for sent in doc.sentences for token in sent.tokens])
                dialog_temp['summary'] = summary_temp
            else:
                # the relevant dialogue is the summary
                dialog_temp['summary'] = [s.strip('#') for s in l['relevant_dialogue']] # remove the # because it is also removed from the training

            # # check if sth goes wrong
            # for s in dialog_temp['summary']:
            #     if s not in dialog_temp['text']:
            #         print(f"The summary of the sentence '{s}' is not in text {dialog_temp['text']}")

            # dialog_temp['finegrained_relevant_dialogue']: I don't think this model can work with this

            # write the jsonl file

            outfile.write(json.dumps(dialog_temp)+'\n')

def prepare_data(tokens=False, force=False, val_split=0.2):
    """Prepare the data for the memsum model.
    
    Args:
        tokens (bool, optional): If True, the text is tokenized in words. Defaults to False, i.e. sentences.
        force (bool, optional): If True, the data is re-preprocessed. Defaults to False.
        
    Returns:
        path_data_folder (os.path): The path to the folder containing the data.
    """
    # go into the memsum folder
    os.chdir(os.path.join(config.ENV_PATH,'MemSum-015ddda'))
    print(os.getcwd())

    from src.data_preprocessing.MemSum.utils import greedy_extract

    # preprocessing custom data
    memsum_datasets = [config.name_trainset, config.name_valset, config.name_testset]

    subprocess.run(["mkdir", "-p", "data"])

    # prepare the data
    for d in memsum_datasets:
        dataset_path = os.path.join(config.ENV_PATH, 'MemSum-015ddda/', config.DATA_PATH, f"memsum_{'tokens' if tokens else 'sentences'}_{d}.jsonl")
        if not os.path.isfile(dataset_path) or force:
            preprocessing_memsum_data(d, tokens)

            # data_path = f"../data/memsum_sentences_{d}.jsonl"

            # subprocess.run(["cp", data_path, "data/"])

            # for the training data make high-ROUGE episodes
            if d == config.name_trainset:
                # compute the indices of the sentences to extract
                train_corpus = [ json.loads(line) for line in open(dataset_path) ]
                for data in tqdm(train_corpus):
                    high_rouge_episodes = greedy_extract(data["text"], data["summary"], beamsearch_size = 2)
                    indices_list = []
                    score_list  = []

                    for indices, score in high_rouge_episodes:
                        indices_list.append( indices )
                        score_list.append(score)

                    data["indices"] = indices_list
                    data["score"] = score_list

                # split the train data in train and validation train
                validation_idxs = np.random.choice(len(train_corpus), size=int(len(train_corpus)*val_split), replace=False)

                # save the data
                dataset_labelled_path = os.path.join(config.ENV_PATH, 'MemSum-015ddda/', config.DATA_PATH, f"memsum_{'tokens' if tokens else 'sentences'}_train_labelled.jsonl")
                dataset_validation_labelled_path = os.path.join(config.ENV_PATH, 'MemSum-015ddda/', config.DATA_PATH, f"memsum_{'tokens' if tokens else 'sentences'}_trainval_labelled.jsonl")
                
                with open(dataset_validation_labelled_path, "w") as v:
                    with open(dataset_labelled_path,"w") as t:
                        for i, data in enumerate(train_corpus):
                            if i in validation_idxs:
                                v.write(json.dumps(data) + "\n")
                            t.write(json.dumps(data) + "\n")

    return os.path.dirname(dataset_path)

def evaluate_memsum(model_folder_path, memsum_val_corpus_path, p_stop_thres, max_extracted_sentences_per_document):
    from summarizers import MemSum
    from rouge_score import rouge_scorer

    rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=True)

    model_path = os.path.join(model_folder_path, "model_batch_250.pt") # ??? TODO: check this

    memsum_custom_data = MemSum(model_path, "model/glove/vocabulary_200dim.pkl",)
                    # gpu = 0 ,  max_doc_len = 500  )
    # def __init__( self, model_path, vocabulary_path, gpu = None , embed_dim=200, num_heads=8, hidden_dim = 1024, N_enc_l = 2 , N_enc_g = 2, N_dec = 3,  max_seq_len =100, max_doc_len = 500  ):
    # TODO: understand the parameters here

    test_corpus_data = [ json.loads(line) for line in open(memsum_val_corpus_path)]

    # evaluate
    scores = []
    for data in tqdm(test_corpus_data):
        gold_summary = data["summary"]
        extracted_summary = memsum_custom_data.extract([data["text"]], 
                                                        p_stop_thres = p_stop_thres, 
                                                        max_extracted_sentences_per_document = max_extracted_sentences_per_document)[0]
        
        score = rouge_cal.score( "\n".join( gold_summary ), "\n".join(extracted_summary)  )
        scores.append( [score["rouge1"].fmeasure, score["rouge2"].fmeasure, score["rougeLsum"].fmeasure ] )
    
    return np.asarray(scores).mean(axis = 0)

def train_memsum(path_to_memsum_training_data, run_uid,
                 tokens=False,
                 hidden_dim=1024, N_enc_l=3, N_enc_g=3, N_dec=3, 
                 num_of_epochs=100, validate_every=1000, 
                 learning_rate=1e-4, warmup_step=1000, weight_decay=1e-6, dropout_rate=0.1, 
                 moving_average_decay=0.999, p_stop_thres=0.7, apply_length_normalization=1, 
                 max_seq_len=100, max_doc_len=500, max_extracted_sentences_per_document=7,
                 num_heads=8, print_every=100, save_every=500, restore_old_checkpoint=True,
                 n_device=8, batch_size_per_device=16):
    os.chdir(os.path.join(config.ENV_PATH,'MemSum-015ddda'))

    # paths to data
    memsum_training_corpus_path = os.path.join(config.ENV_PATH, 'MemSum-015ddda/', config.DATA_PATH,
                                               f"memsum_{'tokens' if tokens else 'sentences'}_train_labelled.jsonl")
    memsum_trainval_labelled_corpus_path = os.path.join(config.ENV_PATH, 'MemSum-015ddda/', config.DATA_PATH,
                                                        f"memsum_{'tokens' if tokens else 'sentences'}_trainval_labelled.jsonl")
    memsum_val_corpus_path = os.path.join(config.ENV_PATH, 'MemSum-015ddda/', config.DATA_PATH,
                                          f"memsum_{'tokens' if tokens else 'sentences'}_test.jsonl")

    # model and log folder
    run_uid = run_uid
    model_folder_path = os.path.join(config.ENV_PATH, 'MemSum-015ddda/', config.MODEL_PATH, 'run/', f"{run_uid}/")
    log_folder_path = os.path.join(config.ENV_PATH, 'MemSum-015ddda/', config.MODEL_PATH, 'run/', f"{run_uid}/", "log/")

    if not os.path.isdir(model_folder_path) or not os.path.isdir(log_folder_path):
        os.makedirs(model_folder_path)
        os.makedirs(log_folder_path)
    else:
        raise(f"Model folder {model_folder_path} or log folder {log_folder_path} already exists. Please choose a different run_uid.")

    # run training
    result = subprocess.run(["cd", str(path_to_memsum_training_data)],
                            capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    result = subprocess.run(["cd", str(path_to_memsum_training_data), "&&",
                             "python3", "train.py", "-training_corpus_file_name", memsum_training_corpus_path, "-validation_corpus_file_name", memsum_trainval_labelled_corpus_path,
                             "-model_folder", str(model_folder_path) , "-log_folder", str(log_folder_path),
                             "-vocabulary_file_name", "../../model/glove/vocabulary_200dim.pkl", "-pretrained_unigram_embeddings_file_name", "../../model/glove/unigram_embeddings_200dim.pkl",
                             "-max_seq_len", str(max_seq_len), "-max_doc_len", str(max_doc_len), "-num_of_epochs", str(num_of_epochs), "-save_every", str(save_every),
                             "-n-device", str(n_device), "-batch_size_per_device", str(batch_size_per_device), "-max_extracted_sentences_per_document", str(max_extracted_sentences_per_document),
                             "-moving_average_decay", str(moving_average_decay), "-p_stop_thres", str(p_stop_thres),],
                             capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    
    # evaluate the model
    # rough_score = evaluate_memsum(model_folder_path, memsum_val_corpus_path, p_stop_thres, max_extracted_sentences_per_document)
    
    return model_folder_path, 'test' # rough_score