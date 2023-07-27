import fact_extraction_model.model.bert_two_heads as model_combined

from fact_extraction_model.shared import BASE_MODEL, OUTPUT_DIR

from transformers import AutoTokenizer
import torch
import pandas as pd

model = model_combined.BertForFactAndAnchorClassification.from_pretrained(OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
id2label_anchors = {}
tag2id_facts = {}
id2label_facts = {}
label2id_facts = {}


# Multi-label inference
def get_offsets_and_predicted_tags(example: str, model, tokenizer, threshold=0):
    """
    Get prediction of model on example, using tokenizer
    Args:
      - example (str): The input text
      - model: The span categorizer
      - tokenizer: The tokenizer
      - threshold: The threshold to decide whether the token should belong to the label. Default to 0, which corresponds to probability 0.5.
    Returns:
      - List of (token, tags, offset) for each token.
    """
    # Tokenize the sentence to retrieve the tokens and offset mappings
    raw_encoded_example = tokenizer(example, return_offsets_mapping=True)
    encoded_example = tokenizer(example, return_tensors="pt").to(device)

    # Call the model. The output LxK-tensor where L is the number of tokens, K is the number of classes
    out = model(**encoded_example)["logits"][0]

    # We assign to each token the classes whose logit is positive
    predicted_tags = [
        [i for i, l in enumerate(logit) if l > threshold] for logit in out
    ]

    return [
        {"token": token, "tags": tag, "offset": offset}
        for (token, tag, offset) in zip(
            tokenizer.batch_decode(raw_encoded_example["input_ids"]),
            predicted_tags,
            raw_encoded_example["offset_mapping"],
        )
    ]


def get_tagged_groups(example: str, model, tokenizer):
    """
    Get prediction of model on example, using tokenizer
    Returns:
    - List of spans under offset format {"start": ..., "end": ..., "tag": ...}, sorted by start, end then tag.
    """
    offsets_and_tags = get_offsets_and_predicted_tags(example, model, tokenizer)
    predicted_offsets = {l: [] for l in tag2id_facts}
    last_token_tags = []
    for item in offsets_and_tags:
        (start, end), tags = item["offset"], item["tags"]

        for label_id in tags:
            label = id2label_facts[label_id]
            tag = label[2:]  # "I-PER" => "PER"
            if label.startswith("B-"):
                predicted_offsets[tag].append({"start": start, "end": end})
            elif label.startswith("I-"):
                # If "B-" and "I-" both appear in the same tag, ignore as we already processed it
                if label2id_facts[f"B-{tag}"] in tags:
                    continue

                if (
                    label_id not in last_token_tags
                    and label2id_facts[f"B-{tag}"] not in last_token_tags
                ):
                    predicted_offsets[tag].append({"start": start, "end": end})
                else:
                    predicted_offsets[tag][-1]["end"] = end

        last_token_tags = tags

    flatten_predicted_offsets = [
        {**v, "tag": k, "text": example[v["start"] : v["end"]]}
        for k, v_list in predicted_offsets.items()
        for v in v_list
        if v["end"] - v["start"] >= 3
    ]
    flatten_predicted_offsets = sorted(
        flatten_predicted_offsets,
        key=lambda row: (row["start"], row["end"], row["tag"]),
    )
    return flatten_predicted_offsets


example = "Du coup, la menace des feux de forêt est permanente, après les incendies dévastateurs de juillet dans le sud-ouest de la France, en Espagne, au Portugal ou en Grèce. Un important feu de forêt a éclaté le 24 juillet dans le parc national de la Suisse de Bohême, à la frontière entre la République tchèque et l'Allemagne, où des records de chaleur ont été battus (36,4C). Un millier d'hectares ont déjà été touchés. Lundi, les pompiers espéraient que l'incendie pourrait être maîtrisé en quelques jours."
print(example)
get_tagged_groups(example, model, tokenizer)


# Single-class inference


def align_tokens_and_predicted_labels(toks_cpu, preds_cpu):
    aligned_toks, aligned_preds = [], []
    prev_tok = None
    for tok, pred in zip(toks_cpu, preds_cpu):
        if tok.startswith("##") and prev_tok is not None:
            prev_tok += tok[2:]
        else:
            if prev_tok is not None:
                aligned_toks.append(prev_tok)
                aligned_preds.append(id2label_anchors[prev_pred])
            prev_tok = tok
            prev_pred = pred
    if prev_tok is not None:
        aligned_toks.append(prev_tok)
        aligned_preds.append(id2label_anchors[prev_pred])
    return aligned_toks, aligned_preds


def predict(texts):
    aligned_tok_list, aligned_pred_list = [], []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(**inputs)
        tokens_cpu = tokenizer.convert_ids_to_tokens(inputs.input_ids.view(-1))
        preds_cpu = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

        aligned_toks, aligned_preds = align_tokens_and_predicted_labels(
            tokens_cpu, preds_cpu
        )

        aligned_tok_list.append(aligned_toks)
        aligned_pred_list.append(aligned_preds)

    return aligned_tok_list, aligned_pred_list


predicted_tokens, predicted_tags = predict(
    [
        ["Sie klagte über anhaltende Müdigkeit, Gewichtszunahme und trockene Haut ."],
        [
            "In ihrer Krankengeschichte ist bekannt, dass sie an einer Schilddrüsenunterfunktion leidet und bereits mit Levothyroxin behandelt wird ."
        ],
    ]
)

pd.DataFrame(
    [predicted_tokens[0], predicted_tags[0]], index=["tokens", "predicted_tags"]
)
pd.DataFrame(
    [predicted_tokens[1], predicted_tags[1]], index=["tokens", "predicted_tags"]
)
