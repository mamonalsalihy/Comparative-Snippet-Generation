import json, os

input_folder = "./review_data/segments/"
output_file = "./qt/data/json/qt_train.json"

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

with open(output_file, "w") as out_f:
    out_f.write(json.dumps(data_list))