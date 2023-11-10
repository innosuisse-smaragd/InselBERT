import pandas as pd
import torch
import shared.model_helpers as helper

import fact_extraction_model.model.bert_two_heads as model_combined
import shared.schema_generator

tokenizer = helper.getTokenizer()
device = helper.getDevice()
model = helper.getPretrainedModel(model_combined)
model = model.to(device)
schema = shared.schema_generator.SchemaGenerator()
num_labels = NUM_LABELS_FACTS_ANCHORS = len(schema.label2id_anchors)

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
    out = model(**encoded_example)["facts"]["logits"][0]

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
    predicted_offsets = {l: [] for l in schema.tag2id_facts}
    last_token_tags = []
    for item in offsets_and_tags:
        (start, end), tags = item["offset"], item["tags"]

        for label_id in tags:
            label = schema.id2label_facts[label_id]
            tag = label[2:]  # "I-PER" => "PER"
            if label.startswith("B-"):
                predicted_offsets[tag].append({"start": start, "end": end})
            elif label.startswith("I-"):
                # If "B-" and "I-" both appear in the same tag, ignore as we already processed it
                if schema.label2id_facts[f"B-{tag}"] in tags:
                    continue

                if (
                    label_id not in last_token_tags
                    and schema.label2id_facts[f"B-{tag}"] not in last_token_tags
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


example = "MAMMOGRAFIE BEIDSEITS IN ZWEI EBENEN VOM 22.06.2021&#10;&#10;Fragestellung/Indikation&#10;Met. Adeno-Ca, unkl. Primarius. (CT-Befund vom 10.6.21 mit Vd. a. i.e.L. DD HCC, DD CCC.&#10;Positive Familienanamnese für Brustkrebs (Tante väterlicherseits).&#10;Malignität/Auffälligkeiten?&#10;&#10;Klinische Untersuchung und Ultraschall:&#10;Durchführung in der gynäkologischen Senologie dieser Klinik (Bericht siehe dort).&#10;&#10;Befund&#10;Zum Vergleich liegt die Voruntersuchung vom 28.09.2020 vor.&#10;&#10;Mammografie beidseits MLO und CC:&#10;Kutis und Subkutis unauffällig.&#10;Mittelfleckiges, teilweise involutiertes Drüsenparenchym beidseits.&#10;Kein malignomsuspekter Herdbefund. Keine suspekte Mikrokalkgruppe.&#10;&#10;Beurteilung&#10;Mammographisch kein Anhalt für Malignität beidseits.&#10;&#10;ACR-Typ b beidseits.&#10;BIRADS 1 beidseits.&#10;&#10;Falls auch der Ultraschallbefund der Brust unauffällig sein sollte, wäre eine weitere Befundabsicherung mittels MR Mammographie zu erwägen."
print(example)
print(get_tagged_groups(example, model, tokenizer))

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
                aligned_preds.append(schema.id2label_anchors[prev_pred])
            prev_tok = tok
            prev_pred = pred
    if prev_tok is not None:
        aligned_toks.append(prev_tok)
        aligned_preds.append(schema.id2label_anchors[prev_pred])
    return aligned_toks, aligned_preds


def predict(texts):
    aligned_tok_list, aligned_pred_list = [], []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = model(**inputs)
        tokens_cpu = tokenizer.convert_ids_to_tokens(inputs.input_ids.view(-1))
        preds_cpu = torch.argmax(outputs["anchors"].logits, dim=-1)[0].cpu().numpy()

        aligned_toks, aligned_preds = align_tokens_and_predicted_labels(
            tokens_cpu, preds_cpu
        )

        aligned_tok_list.append(aligned_toks)
        aligned_pred_list.append(aligned_preds)

    return aligned_tok_list, aligned_pred_list


predicted_tokens, predicted_tags = predict(
    [
        [
           example
        ],
    ]
)


result = pd.DataFrame(
    [predicted_tokens[0], predicted_tags[0]], index=["tokens", "predicted_tags"]
)
result.to_csv("./result_anchors.csv")
print(result)
