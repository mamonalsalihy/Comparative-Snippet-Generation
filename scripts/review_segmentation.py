"""
1. Sentence segmentation
2. Segment sentiment classification
3. Aggregate segments into one representative segment
4. Join representative segments together
5. Gradio for front-end
"""
import argparse
import collections

import requests as requests
import openai
import re, spacy, string
import os
import pickle
import json
import pandas as pd
from tqdm import tqdm

import en_core_web_sm

nlp = en_core_web_sm.load()

from spacy.symbols import PRON, PROPN, NOUN, AUX, VERB, DET, ADP, ADJ


def pre_process_and_validation_check(segment):
    token_list = list(nlp(segment))

    verb_list = [AUX, VERB]
    noun_list = [NOUN, PRON, PROPN]
    first_word_pos = [NOUN, PRON, PROPN, DET, ADP, ADJ]

    first_token_list = ["because", "and", "before", "but", "however", "now", "of", "then", "&", "or", "that",
                        "although"]
    first_person_and_pronoun_words = ["i", "me", "my", "myself", "mine", "we", "us", "our", "ourselves", "he", "she",
                                      "him", "her", "am", "they", "them", "themselves", "their",
                                      "theirs",
                                      "you", "yours", "your"]
    first_word_skip_segment = ["who", "what", "when", "whom", "how", "whenever", "where", "which", "whose", "why",
                               "after",
                               "hey", "to", "in", "if", "so"]

    # remove leading and trailing punctuations.
    # if first token is in `first_token_list`, remove it.
    while (len(token_list) > 0 and ((token_list[0].text in string.punctuation) or
                                    (token_list[0].text in first_token_list))):
        if len(token_list) == 1:
            token_list = []
        else:
            token_list = token_list[1:]

    while len(token_list) > 0 and token_list[-1].text in string.punctuation:
        if len(token_list) == 1:
            token_list = []
        else:
            token_list = token_list[:-1]

    # if number of tokens are less than 2, skip the segment
    if len(token_list) <= 2:
        return None, None

    # if first word is a verb, skip the segment
    if token_list[0].pos in verb_list:
        return None, None

    # if first word in first_word_skip_segment list, skip the segment
    if token_list[0].text in first_word_skip_segment:
        return None, None

    # If first person word in a segment, skip the segment
    is_found = False
    for token in token_list:
        if token.text in first_person_and_pronoun_words:
            is_found = True
            break

    if is_found:
        return None, None

    # Check whether a segment has at least one noun and one verb
    is_noun = False
    is_verb = False

    tmp_list = []

    for token in token_list:
        tmp_list.append(token.pos_)
        if token.pos in noun_list:
            is_noun = True
        elif token.pos in verb_list:
            is_verb = True
    if (not is_noun) or (not is_verb):
        return None, None

    return token_list, tuple(tmp_list)


class Segmentation:
    def __init__(self, folder_path, output_file_pattern):
        self.folder_path = folder_path
        self.output_file_pattern = output_file_pattern
        segment_dir = output_file_pattern.split('/')[-2]
        if not os.path.isdir(segment_dir):
            os.mkdir(output_file_pattern[:-3])

        self.postprocess_by_file()

    def postprocess_by_file(self):
        for file in tqdm(os.listdir(self.folder_path)):
            segment_file = file[:7] + "segment_" + file[7:]
            folder_path_pattern = self.folder_path + '/{}'
            self.postprocess_segments(folder_path_pattern.format(file),
                                      self.output_file_pattern.format(segment_file))

        print("Finished cleaning all review segments")

    @staticmethod
    def postprocess_segments(file_path, output_path):
        pattern_2 = re.compile(r"-[A-Z]{3}-")

        with open('unique_spacy_pos_pattern.pkl', 'rb') as fd:
            pattern_set = pickle.load(fd)['pattern']

        with open(file_path, encoding='utf8') as fd:
            for line in fd:
                if 'eos-eos' in line or len(line) < 4:
                    continue
                segments = line.split("EDU_BREAK")
                for segment in segments:
                    segment = pattern_2.sub("", segment)
                    segment = segment.lower()
                    segment = " ".join(segment.split())
                    token_list, pos_tuple = pre_process_and_validation_check(segment)
                    if token_list:
                        segment = " ".join([token.text for token in token_list])
                        if pos_tuple in pattern_set and len(segment) == len(segment.encode('utf8')):
                            with open(output_path, 'a') as f_out:
                                f_out.write(segment)
                                f_out.write("\n")


def main():
    parser = argparse.ArgumentParser()

    segmentation_parser = parser.add_argument_group("Segmentation")
    segmentation_parser.add_argument('--folder_path', type=str, default='../review_data/edus')
    segmentation_parser.add_argument('--output_file_pattern', type=str, default='../review_data/segments/{}')

    args = parser.parse_args()

    Segmentation(args.folder_path, args.output_file_pattern)


if __name__ == '__main__':
    main()
