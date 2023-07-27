def get_tag2id(entity):
    if entity == "facts":
        print("getting facts")
    elif entity == "anchors":
        print("getting anchors")
    elif entity == "modifiers":
        print("modifiers")


def convert_tags_to_labels(tag2id: dict):
    return {
        "O": 0,
        **{f"B-{k}": 2 * v - 1 for k, v in tag2id.items()},
        **{f"I-{k}": 2 * v for k, v in tag2id.items()},
    }


def inverse_2id(mapping: dict):
    return {v: k for k, v in mapping.items()}
