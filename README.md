# LLM-based, two-layer clinical fact extraction based on the approach by Steinkamp et al.

## Repo set-up

- Install pdm on your machine: https://pdm.fming.dev/latest/
- cd into project root
- run `pdm install`
- A venv should be created in the project root and activated automatically. If not, [activate it manually](https://pdm-project.org/latest/usage/venv/#activate-a-virtualenv).

## Available scripts

In the file `pyproject.toml`, several scripts are defined that can be run via `pdm <script-name>`. Do not run scripts directly using `python <script-name>` to avoid issues.

## Description of serialized models (contained in `./serialized_models`)

### medbert_512

This model serves as the basis for further development. The folder contains the git-repo of the model itself. More information .

### medbert_512_pretrained

This folder contains the result of the further-pretraining of [MedBERT.de](https://huggingface.co/GerMedBERT/medbert-512) on the Insel corpus based on Masked Language Modelling (MLM).

### inselBERT_extraction_finetuned

This folder contains the fine-tuned model which is composed of the medbert_512_pretrained encoder body and two classification heads (one single-label, one multi-label). The heads predict for each token to which facts (0..n) it might belong and whether the token corresponds to an anchor entity, respectively. This architecture corresponds to the method proposed by Steinkamp et al. Once trained, this model takes a clinical text as input and returns annotations for each token (facts and anchors).

### inselBERT_extraction_finetuned_refined

This folder contains the second, fine-tuned model which is composed of the same medbert_512_pretrained encoder body of medbert_512_facts. Additionally, three classification heads are added. Two binary classification heads predict whether a token is the beginning or end of a fact span, respectively. A third, multi-class classification head predicts which modifier (0..1) a token is part of. Once trained, this model takes a fact candidate, which corresponds to the validated output of the former model, X tokens before and after the fact candidate as well as the information which fact was identified (see open issues).

### inselbert_qa_hf

This folder contains the model for extractive question answering trained with the example script provided by HuggingFace.
The model must be trained in a separate repository, as source installation of the transformers library is required. 
First, generate training data running `pdm generate-dataset-json-for-qa-model`. Copy the folder "./data/protected/qa_json" into the other repository and train the model. 
Copy the resulting model files into ./serialized_models/inselbert_qa_hf and run `pdm save-qa-model-to-bento` to create a BentoService for the model.
The model takes a question and a clinical text as input and returns the answer span to the question.

## Open issues

- Context information: How can we integrate context information (which fact was identified, position of anchor, position of fact within wider context) into the model?
- Fact candidate calculation: Where and how are fact candidates calculated? Is this part of one of the two models?
- Sharing of modifiers: What impact does sharing of modifiers have on performance?
- Ignore special tokens: Multi-label classification does not automatically ignore -100 tokens when calculating loss. 
- Merging facts: What impact does merging of facts with the same modifiers have on model performance and structuring? 
