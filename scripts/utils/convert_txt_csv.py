import pandas as pd


def main():
    result = []
    with open('../spot-data/data/spot-yelp13-edus.txt') as fd:
        for line in fd:
            data = {}
            line = line.split()
            if line:
                text = " ".join(line[1:])
                if (len(text) == 7 and text.isnumeric()) or line[0] == '0':
                    continue
                if line[0] == '-':
                    data['labels'] = 0
                if line[0] == '+':
                    data['labels'] = 1
                if line[-1] == '<s>':
                    line = line[1:-1]
                    data['text'] = " ".join(line)
                    continue
                if len(text) > 1:
                    data['text'] = text.replace(",", "")
                result.append(data)

    df = pd.DataFrame(result)
    df.to_csv('./spot-data/data/spot-yelp13-edus.csv', index=False)


if __name__ == '__main__':
    main()
