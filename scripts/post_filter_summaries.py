import argparse

from lemminflect import getLemma
import spacy, os

en_spacy = spacy.load("en_core_web_sm")


def sort_by_length(args):
    for file in os.listdir(args.input_folder):
        if args.sentiment == 'positive':
            path = args.output_folder_pos
        else:
            path = args.output_folder_neg
        with open(os.path.join(path, file), "w+") as out_f:
            with open(os.path.join(args.input_folder, file), "r") as in_f:
                line_list = []
                for line in in_f:
                    word_list = []
                    try:
                        for token in en_spacy(line.strip()):
                            if token.tag_ == "VBD" or \
                                    token.tag_ == "VBN" or \
                                    token.tag_ == "MD":
                                word_list.append(getLemma(token.text, upos="VERB")[0])
                            else:
                                word_list.append(token.text)
                    except:
                        continue
                    line_list.append((len(word_list), " ".join(word_list)))

                sorted_line_list = sorted(line_list, key=lambda x: x[0], reverse=True)

                for elem in sorted_line_list:
                    out_f.write(elem[1] + "\n")


def filter_by_last_word(args):
    if args.sentiment == 'positive':
        # for positive opinions
        tag_list = ["TO", "LS", "WDT", "PRP$", "XX", ":", "VBZ", "CD", "PRP", "DT",
                    "IN", "VBP", "''", "FW", "NFP", "WRB", "RBS", "WP"]
        last_word_list = ["and", "but"]

        for file in os.listdir(args.last_word_filter_input_folder_pos):
            with open(os.path.join(args.last_word_filter_output_folder_pos, file), "w+") as out_f:
                with open(os.path.join(args.last_word_filter_input_folder_pos, file), "r") as in_f:
                    for line in in_f:
                        token_list = en_spacy(line.strip())

                        if token_list[-1].tag_ in tag_list:
                            continue
                        elif len(token_list) <= 3:
                            continue
                        elif token_list[-1].tag_ == "CC":
                            if token_list[-1].text in last_word_list:
                                token_list = token_list[:-1]
                            elif token_list[-1].text == "plus":
                                continue

                        out_f.write(" ".join([token.text for token in token_list]) + "\n")
    else:
        # for negative opinions
        tag_list = ["TO", "LS", "ADD", "WDT", "PRP$", "XX", ":", "VBZ", "CD", "PRP", "DT",
                    "IN", "NNP", "VBP", "''", "FW", "NFP", "WRB", "RBS", "WP"]
        last_word_list = ["and", "but", "plus", "or", "neither"]

        for file in os.listdir(args.last_word_filter_input_folder_neg):
            with open(os.path.join(args.last_word_filter_output_folder_neg, file), "w") as out_f:
                with open(os.path.join(args.last_word_filter_input_folder_neg, file), "r") as in_f:
                    for line in in_f:
                        token_list = en_spacy(line.strip())

                        if token_list[-1].tag_ in tag_list:
                            continue
                        elif len(token_list) <= 3:
                            continue
                        elif token_list[-1].tag_ == "CC":
                            if token_list[-1].text in last_word_list:
                                token_list = token_list[:-1]

                        out_f.write(" ".join([token.text for token in token_list]) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default='../qt/outputs/general_run1')
    parser.add_argument("--output_folder_pos", type=str, default='../review_data/pos_sum/summaries_sort_by_length')
    parser.add_argument("--output_folder_neg", type=str, default='../review_data/neg_sum/summaries_sort_by_length')
    parser.add_argument("--last_word_filter_input_folder_pos", type=str, default='../review_data/pos_sum'
                                                                                 '/summaries_sort_by_length')
    parser.add_argument("--last_word_filter_output_folder_pos", type=str, default='../review_data/pos_sum'
                                                                                  '/summaries_last_word_filter')
    parser.add_argument("--last_word_filter_input_folder_neg", type=str,
                        default='../review_data/neg_sum/summaries_sort_by_length')
    parser.add_argument("--last_word_filter_output_folder_neg", type=str,
                        default='../review_data/neg_sum/summaries_last_word_filter')
    parser.add_argument("--sentiment", type=str, default='negative')
    args = parser.parse_args()

    if not os.path.isdir(args.output_folder_pos):
        os.makedirs(args.output_folder_pos, exist_ok=True)
        print("{} was created".format(args.output_folder_pos))

    if not os.path.isdir(args.output_folder_neg):
        os.makedirs(args.output_folder_neg, exist_ok=True)
        print("{} was created".format(args.output_folder_neg))

    if not os.path.isdir(args.last_word_filter_input_folder_pos):
        os.makedirs(args.last_word_filter_input_folder_pos, exist_ok=True)
        print("{} was created".format(args.last_word_filter_input_folder_pos))

    if not os.path.isdir(args.last_word_filter_input_folder_neg):
        os.makedirs(args.last_word_filter_input_folder_neg, exist_ok=True)
        print("{} was created".format(args.last_word_filter_input_folder_neg))

    if not os.path.isdir(args.last_word_filter_output_folder_pos):
        os.makedirs(args.last_word_filter_output_folder_pos, exist_ok=True)
        print("{} was created".format(args.last_word_filter_output_folder_pos))

    if not os.path.isdir(args.last_word_filter_output_folder_neg):
        os.makedirs(args.last_word_filter_output_folder_neg, exist_ok=True)
        print("{} was created".format(args.last_word_filter_output_folder_neg))

    sort_by_length(args)
    filter_by_last_word(args)


if __name__ == '__main__':
    main()
