import json, os
import argparse


def convert_to_json(input_folder, output_file):
    data_list = []

    for file in os.listdir(input_folder):
        entity_dict = {"entity_id": file, "entity_name": file, "reviews": []}
        id_counter = 0
        counter = 1
        tmp_dict = {"review_id": str(id_counter), "sentences": []}

        with open(os.path.join(input_folder, file), "r") as in_f:
            for line in in_f:
                tmp_dict["sentences"].append(line.strip())

                if counter % 5 == 0:
                    entity_dict["reviews"].append(tmp_dict)
                    id_counter = id_counter + 1
                    tmp_dict = {"review_id": str(id_counter), "sentences": []}

                counter = counter + 1

        data_list.append(entity_dict)

    with open(output_file, "w+") as out_f:
        out_f.write(json.dumps(data_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default="../review_data/segments/")
    parser.add_argument('--output_file', type=str, default="../qt/data/json/qt_train.json")

    args = parser.parse_args()

    convert_to_json(args.input_folder, args.output_file)


if __name__ == '__main__':
    main()
