def convert_iob2_to_dataset(iob2_file):
    dataset = []
    sentence_id = 0
    tokens = []
    ner_tags = []
    tag_to_label = {}
    label_counter = 0

    with open(iob2_file, "r") as file:
        for line in file:
            line = line.strip()

            if not line:
                if tokens:
                    data = {
                        "id": str(sentence_id),
                        "ner_tags": ner_tags,
                        "tokens": tokens,
                    }
                    dataset.append(data)
                    sentence_id += 1
                    tokens = []
                    ner_tags = []
            else:
                parts = line.split()
                token = parts[0]
                tag = parts[-1]

                if tag not in tag_to_label:
                    tag_to_label[tag] = label_counter
                    label_counter += 1

                label = tag_to_label[tag]

                tokens.append(token)
                ner_tags.append(label)

    if tokens:
        data = {"id": str(sentence_id), "ner_tags": ner_tags, "tokens": tokens}
        dataset.append(data)

    return dataset, tag_to_label
