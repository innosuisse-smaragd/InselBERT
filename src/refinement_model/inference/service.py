import bentoml
import pandas as pd
import torch

import constants
import shared.model_helpers as helper

model_ref = bentoml.models.get(constants.FACT_EXTRACTION_MODEL_NAME + ":latest")


class FactExtractionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    tokenizer = helper.getTokenizer()
    device = "cpu"

    def __init__(self):
        self.model = bentoml.transformers.load_model(model_ref)
        self.schema = model_ref.custom_objects.get("fact_schema")
        self.num_labels = len(self.schema.label2id_anchors)

    @staticmethod
    def get_offsets_and_predicted_tags(example: str, tokenizer, outputs, threshold=0):
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

        out = outputs["facts"]["logits"][0]
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

    # Single-class inference

    def align_tokens_and_predicted_labels(self, toks_cpu, preds_cpu):
        aligned_toks, aligned_preds = [], []
        prev_tok = None
        for tok, pred in zip(toks_cpu, preds_cpu):
            if tok.startswith("##") and prev_tok is not None:
                prev_tok += tok[2:]
            else:
                if prev_tok is not None:
                    aligned_toks.append(prev_tok)
                    aligned_preds.append(self.schema.id2label_anchors[prev_pred])
                prev_tok = tok
                prev_pred = pred
        if prev_tok is not None:
            aligned_toks.append(prev_tok)
            aligned_preds.append(self.schema.id2label_anchors[prev_pred])
        return aligned_toks, aligned_preds

    def predict_anchors(self, tokenizer, tokenized_input, outputs): # TODO: Remove alignment
        tokens_cpu = tokenizer.convert_ids_to_tokens(tokenized_input.input_ids.view(-1))
        preds_cpu = torch.argmax(outputs["anchors"].logits, dim=-1)[0].cpu().numpy()
        converted_preds_cpu = []
       # aligned_toks, aligned_preds = self.align_tokens_and_predicted_labels(
        #    tokens_cpu, preds_cpu
       # )
        for pred in preds_cpu:
            converted_preds_cpu.append(self.schema.id2label_anchors[pred])

        return tokens_cpu, converted_preds_cpu

    def predict_facts(self,example, tokenizer, outputs):
        """
        Get prediction of model on example, using tokenizer
        Returns:
        - List of spans under offset format {"start": ..., "end": ..., "tag": ...}, sorted by start, end then tag.
        """
        offsets_and_tags = self.get_offsets_and_predicted_tags(example, tokenizer, outputs)
        print("Raw facts: ", offsets_and_tags)
        predicted_offsets = {l: [] for l in self.schema.tag2id_facts}
        last_token_tags = []
        for item in offsets_and_tags:
            (start, end), tags = item["offset"], item["tags"]

            for label_id in tags:
                label = self.schema.id2label_facts[label_id]
                tag = label[2:]  # "I-PER" => "PER"
                if label.startswith("B-"):
                    predicted_offsets[tag].append({"start": start, "end": end})
                elif label.startswith("I-"):
                    # If "B-" and "I-" both appear in the same tag, ignore as we already processed it
                    if self.schema.label2id_facts[f"B-{tag}"] in tags:
                        continue

                    if (
                            label_id not in last_token_tags
                            and self.schema.label2id_facts[f"B-{tag}"] not in last_token_tags
                    ):
                        predicted_offsets[tag].append({"start": start, "end": end})
                    else:
                        predicted_offsets[tag][-1]["end"] = end

            last_token_tags = tags

        flatten_predicted_offsets = [
            {**v, "tag": k, "text": example[v["start"]: v["end"]]}
            for k, v_list in predicted_offsets.items()
            for v in v_list
            if v["end"] - v["start"] >= 3
        ]
        flatten_predicted_offsets = sorted(
            flatten_predicted_offsets,
            key=lambda row: (row["start"], row["end"], row["tag"]),
        )
        return flatten_predicted_offsets

    @bentoml.Runnable.method(batchable=False)
    def predict_facts_and_anchors(self,example: str):
        tokenized_input = self.tokenizer(example, return_tensors="pt").to(self.device)
        outputs = self.model(**tokenized_input)
        predicted_anchor_tokens, predicted_anchor_tags = self.predict_anchors(self.tokenizer, tokenized_input, outputs)
        structured_anchors_dataframe = pd.DataFrame(
            [predicted_anchor_tokens, predicted_anchor_tags], index=["tokens_anchors", "predicted_anchor_tags"]
        )
        #print("Anchors_dataframe: ", structured_anchors_dataframe)

        # Multi.label inference for facts:
        raw_fact_predictions = self.get_offsets_and_predicted_tags(example, self.tokenizer, outputs)
        #print("Raw fact predictions: ", raw_fact_predictions)
        extracted_fact_predictions = []
        extracted_fact_tokens = []
        for pred in raw_fact_predictions:
            if len(pred["tags"]) > 0:
                extracted_fact_predictions.append(pred["tags"])
            else:
                extracted_fact_predictions.append('O')
            extracted_fact_tokens.append(pred["token"])
        facts_dataframe = pd.DataFrame([extracted_fact_tokens, extracted_fact_predictions],
                                       index=["tokens_facts", "predicted_fact_tags"])
        merged_dataframe = pd.concat([structured_anchors_dataframe, facts_dataframe])
        if extracted_fact_tokens != predicted_anchor_tokens:
            raise Exception("Anchor and fact tokens do not match!")
        return merged_dataframe


fact_extraction_runner = bentoml.Runner(FactExtractionRunnable, name="fact_extraction_runner", models=[model_ref])
svc = bentoml.Service("extract_bert", runners=[fact_extraction_runner])


@svc.api(input=bentoml.io.Text(), output=bentoml.io.PandasDataFrame())
async def extract_facts_and_anchors(inp: str):
    return await fact_extraction_runner.predict_facts_and_anchors.async_run(inp)

if __name__ == "__main__":
    example = "MAMMOGRAFIE BEIDSEITS IN ZWEI EBENEN VOM 22.06.2021&#10;&#10;Fragestellung/Indikation&#10;Met. Adeno-Ca, unkl. Primarius. (CT-Befund vom 10.6.21 mit Vd. a. i.e.L. DD HCC, DD CCC.&#10;Positive Familienanamnese für Brustkrebs (Tante väterlicherseits).&#10;Malignität/Auffälligkeiten?&#10;&#10;Klinische Untersuchung und Ultraschall:&#10;Durchführung in der gynäkologischen Senologie dieser Klinik (Bericht siehe dort).&#10;&#10;Befund&#10;Zum Vergleich liegt die Voruntersuchung vom 28.09.2020 vor.&#10;&#10;Mammografie beidseits MLO und CC:&#10;Kutis und Subkutis unauffällig.&#10;Mittelfleckiges, teilweise involutiertes Drüsenparenchym beidseits.&#10;Kein malignomsuspekter Herdbefund. Keine suspekte Mikrokalkgruppe.&#10;&#10;Beurteilung&#10;Mammographisch kein Anhalt für Malignität beidseits.&#10;&#10;ACR-Typ b beidseits.&#10;BIRADS 1 beidseits.&#10;&#10;Falls auch der Ultraschallbefund der Brust unauffällig sein sollte, wäre eine weitere Befundabsicherung mittels MR Mammographie zu erwägen."
    runnable = FactExtractionRunnable()
    result = runnable.predict_facts_and_anchors(example)
    print(result)
