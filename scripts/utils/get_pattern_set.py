import pickle

def main():
    """
    1. Remove curly brackets and commas
    2. Add resultant string to set
    :return:
    """
    pattern_set = set()
    with open('pattern_set') as fd:
        for line in fd:
            line = line.strip()
            line = line.replace('\"','')
            if line[0] == '{':
                pattern_set.add(eval(line[1:-1]))
            elif line[-1] == '}' or line[-1] == ',':
                pattern_set.add(eval(line[:-1]))

    output_file = 'unique_spacy_pos_pattern.pkl'
    with open(output_file, "wb") as out_f:
        pickle.dump({"pattern": pattern_set}, out_f)

if __name__ == '__main__':
    main()