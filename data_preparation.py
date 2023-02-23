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
                    'n_sentences_dialogue', 'n_tokens_dialogue', 'n_tokens_per_sentence',  # for the dialogues
                    'n_extracts_per_summary_rel',  'n_tokens_per_extract_rel',
                    'n_extracts_per_summary_fine', 'n_tokens_per_extract_fine']
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
            
            metadata_line = pd.DataFrame({'name': name_dataset,
                                        'n_dialogues': n_dialogues,
                                        'n_sentences_dialogue': n_sentences_dialogue,
                                        'n_tokens_dialogue': n_tokens_dialogue,
                                        'n_tokens_per_sentence': n_tokens_per_sentence,
                                        'n_extracts_per_summary_rel': n_extracts_per_summary_rel,
                                        'n_tokens_per_extract_rel': n_tokens_per_extract_rel,
                                        'n_extracts_per_summary_fine': n_extracts_per_summary_fine,
                                        'n_tokens_per_extract_fine': n_tokens_per_extract_fine}, index=[0])
            
            metadata = pd.concat([metadata, metadata_line],
                                ignore_index=True)

    metadata.to_csv(path_to_metadata, index=False)

def preprocessing_memsum_data_sentences(name_dataset, tokens=False):
    """Takes the data from the json file and creates a jsonl file for the memsum model.

    Args:
        name_dataset (string): The name of the dataset. It is used to find the json file and to name the jsonl file.
        tokens (bool, optional): If True, the text is tokenized. Defaults to False.
    """
    # start the tokenizer
    nlp = stanza.Pipeline(lang='en', processors='tokenize')

    with open(os.path.join(config.ENV_PATH, config.DATA_PATH, name_dataset + '.json')) as f:
        json_data = json.load(f)

    memsum_training = os.path.join(config.ENV_PATH, config.DATA_PATH, str('memsum_sentences_' + name_dataset + f"{'tokens' if tokens else 'sentences'}.jsonl"))
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

            outfile.write(json.dumps(dialog_temp)+'\n')

