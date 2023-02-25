import os
import json
import config
import stanza
import logging
import pandas as pd

def computing_metadata(path_to_metadata):
    """Computes the metadata for the datasets and saves it in a csv file.

    Args:
        path_to_metadata (string): The path to the csv file where the metadata is saved.
    """
    datasets = [ config.name_testset, config.name_trainset, config.name_valset, ]
    columns_frame = ['name', 'n_dialogues',  # global per dataset
                    'n_sentences_dialogue', 'n_tokens_dialogue', 'n_tokens_per_sentence', 'n_tokens_max_per_sentence_dialogue'  # for the dialogues
                    'n_extracts_per_summary_rel',  'n_tokens_per_extract_rel',
                    'n_extracts_per_summary_fine', 'n_tokens_per_extract_fine',]
                    # 'first_sentence']
    metadata = pd.DataFrame(columns = columns_frame)  # for the summaries

    nlp = stanza.Pipeline(lang='en', processors='tokenize')

    logging.basicConfig(filename=str(config.DATA_PATH)+'dialoge_problems.log',
                        filemode='w',
                        level=logging.DEBUG)
    logging.info('Start logging')

    for name_dataset in datasets:
        print(f"Start preprocessing {name_dataset}")
        with open(os.path.join(config.ENV_PATH, config.DATA_PATH, name_dataset + '.json')) as f:
            json_data = json.load(f)

        n_dialogues = len(json_data)
        for l in json_data:
            dialogue = nlp(l['dialogue'])

            n_sentences_dialogue = len(dialogue.sentences)
            n_tokens_dialogue = sum([len(sentence.tokens) for sentence in dialogue.sentences])
            n_tokens_max_per_sentence_dialogue = max([len(sentence.tokens) for sentence in dialogue.sentences])
            n_tokens_per_sentence = n_tokens_dialogue / n_sentences_dialogue

            n_extracts_per_summary_rel = len(l['relevant_dialogue'])
            n_tokens_per_extract_rel = 0
            for j in l['relevant_dialogue']:
                relevant_dialogue = nlp(j)
                n_tokens_per_extract_rel += sum([len(sentence.tokens) for sentence in relevant_dialogue.sentences])
            try:
                n_tokens_per_extract_rel = n_tokens_per_extract_rel / n_extracts_per_summary_rel
            except ZeroDivisionError:
                n_tokens_per_extract_rel = 0
                logging.info(f"For {name_dataset} there is a ZeroDivisionError in relevant dialogue: {l}")

            n_extracts_per_summary_fine = len(l['finegrained_relevant_dialogue'])
            n_tokens_per_extract_fine = 0
            for j in l['finegrained_relevant_dialogue']:
                finegrained_relevant_dialogue = nlp(j)
                n_tokens_per_extract_fine += sum([len(sentence.tokens) for sentence in finegrained_relevant_dialogue.sentences])
            try:
                n_tokens_per_extract_fine = n_tokens_per_extract_fine / n_extracts_per_summary_fine
            except ZeroDivisionError:
                n_tokens_per_extract_fine = 0
                logging.info(f"For {name_dataset} there is a ZeroDivisionError in finegrained relevant dialogue: {l}")
            
            # first_sentence = [ token for token in sentence.tokens for (i, sentence) in enumerate(dialogue.sentences) if i == 0]
            metadata_line = pd.DataFrame({'name': name_dataset,
                                        'n_dialogues': n_dialogues,
                                        'n_sentences_dialogue': n_sentences_dialogue,
                                        'n_tokens_dialogue': n_tokens_dialogue,
                                        'n_tokens_max_per_sentence_dialogue': n_tokens_max_per_sentence_dialogue,
                                        'n_tokens_per_sentence': n_tokens_per_sentence,
                                        'n_extracts_per_summary_rel': n_extracts_per_summary_rel,
                                        'n_tokens_per_extract_rel': n_tokens_per_extract_rel,
                                        'n_extracts_per_summary_fine': n_extracts_per_summary_fine,
                                        'n_tokens_per_extract_fine': n_tokens_per_extract_fine,
                                        # 'first_sentence': first_sentence,
                                        }, index=[0])
            
            metadata = pd.concat([metadata, metadata_line],
                                ignore_index=True)

    metadata.to_csv(path_to_metadata, index=False)