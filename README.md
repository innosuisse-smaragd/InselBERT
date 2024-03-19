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

## Deployment

### Inference API Server: Bentoml
To generate a docker image, follow these steps: 
1) Run `pdm save-qa-model-to-bento` and `pdm save-seq-model-to-bento` to load the serialized models into the local bentoml model store.
   2) Run `pdm build-bento` to package both models into a so-called Bento.
3) Run `pdm containerize-bento` to create a docker image from the Bento.
4) To start a docker container, run `docker run -p 3000:3000 <image-name> bentoml serve --reload`. Notice the `--reload` flag, which is needed due to performance issues in production mode.
5) (Internal deployment: Re-tag image with tag ":inference" and push to gitlab registry, then pull image from portainer)


### UI: Streamlit app 
1) Export dependencies with `pdm export -o requirements.txt --without-hashes` (only if changes were done)
2) Make sure that references to serialized models in `streamlit_app.py` are up-to-date
3) Deployment:
   - Internal: Push to gitlab to trigger the CI/CD pipeline building a docker container
   - External: Run `streamlit run streamlit_app.py` to start the app locally
   - External: Build a docker image with `docker build -t inselbertExtractor .` and run it with `docker run -p 8501:8501 inselbertExtractor` 

### Evaluation

#### Comparison between inselBERT and MedBERT.de

   Metric: McNemar test.

1) Fine-tune two models each for question answering and sequence labelling
2) 

#### (Cross-validation of question answering model)

#### Cross-validation of sequence labelling model

- Train k models by generating folds of training data
- Evaluate all models on the same validation data using HF evaluation
#### Evaluation of question answering model
- Metric: squad_v2 (including no_answer metric)
   - Bootstrap of validation data set 
   - run ... 

#### Evaluation of sequence labelling model
- Metric: seqeval
   - Make sure to have a trained model in the serialized_models folder
   - Bootstrap of validation data set
   - run `pdm evaluate-seq-model` to evaluate the sequence labelling model
Output: 

## Open issues

- Context information: How can we integrate context information (which fact was identified, position of anchor, position of fact within wider context) into the model?
- Performance drop upon production mode: see https://github.com/bentoml/BentoML/issues/2267
- Sharing of modifiers: What impact does sharing of modifiers have on performance?
- Ignore special tokens: Multi-label classification does not automatically ignore -100 tokens when calculating loss. 
- Merging facts: What impact does merging of facts with the same modifiers have on model performance and structuring? 
