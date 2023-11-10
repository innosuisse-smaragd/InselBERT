import bentoml
import pandas as pd
import torch

import constants
import src.shared.model_helpers as helper
#import src.shared.schema_generator as schema_generator

model_ref = bentoml.models.get(constants.FACT_EXTRACTION_MODEL_NAME + ":latest")


class FactExtractionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    tokenizer = helper.getTokenizer()
    device = "cpu"
    #schema = schema_generator.SchemaGenerator()

    #num_labels = NUM_LABELS_FACTS_ANCHORS = len(schema.label2id_anchors)

    def __init__(self):
        self.model = bentoml.transformers.load_model(model_ref)
        self.schema = model_ref.custom_objects.get("fact_schema")

    def align_tokens_and_predicted_labels(self,toks_cpu, preds_cpu):
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

    def predict(self, texts):
        aligned_tok_list, aligned_pred_list = [], []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            tokens_cpu = self.tokenizer.convert_ids_to_tokens(inputs.input_ids.view(-1))
            preds_cpu = torch.argmax(outputs["anchors"].logits, dim=-1)[0].cpu().numpy()

            aligned_toks, aligned_preds = self.align_tokens_and_predicted_labels(
                tokens_cpu, preds_cpu
            )

            aligned_tok_list.append(aligned_toks)
            aligned_pred_list.append(aligned_preds)

        return aligned_tok_list, aligned_pred_list


    @bentoml.Runnable.method(batchable=False)
    def extract_facts_and_anchors(self, inp: str):
        predicted_tokens, predicted_tags = self.predict([[inp]])
        result = pd.DataFrame(
            [predicted_tokens[0], predicted_tags[0]], index=["tokens", "predicted_tags"]
        )
        return result.to_numpy()


fact_extraction_runner = bentoml.Runner(FactExtractionRunnable, name="fact_extraction_runner", models=[model_ref])
svc = bentoml.Service("extract_bert", runners=[fact_extraction_runner])


@svc.api(input=bentoml.io.Text(), output=bentoml.io.NumpyNdarray())
async def extract_facts_and_anchors(inp: str):
    return await fact_extraction_runner.extract_facts_and_anchors.async_run(inp)

