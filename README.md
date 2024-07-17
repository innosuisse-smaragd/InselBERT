# LLM-based, two-layer clinical fact extraction based on the approach by Steinkamp et al.

## Repo set-up

- Install pdm on your machine: https://pdm.fming.dev/latest/
- cd into project root
- run `pdm install`
- A venv should be created in the project root and activated automatically. If not, [activate it manually](https://pdm-project.org/latest/usage/venv/#activate-a-virtualenv).
- To pre-train the model on encrypted data, run 'export DECRYTPION_KEY=<key>' in the terminal.

## Available scripts

In the file `pyproject.toml`, several scripts are defined that can be run via `pdm <script-name>`. Do not run scripts directly using `python <script-name>` to avoid issues.

## Description of resulting models

### inselbert

This folder contains the further-pretrained model variants based on [MedBERT.de](https://huggingface.co/GerMedBERT/medbert-512).
The "ALL"-models are further-pretrained on the whole pretraining corpus. 
The "MAMMO"-models are further-pretrained on the mammography subset of the pretraining corpus for 3 and for 10 epochs.

#### Training
- In constants.py, set the pretraining strategy to "ALL" or "MAMMO" to choose the desired model variant.
- If needed, adapt epoch number in the training file.
- Run `pdm further-pretrain`

### inselbert_qa_hf

This folder contains the models fine-tuned for extractive question answering trained with the example script provided by HuggingFace.
The model must be trained in a separate repository, as source installation of the transformers library is required. 
The final model takes a question and a clinical text as input and returns the answer span to the question.

#### Training
- Clone and set-up [this transformers repo](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/README.md) provided by HuggingFace.
- Copy the further-pretrained model variants from the inselbert folder into the transformers repo.
- Here, generate training data by running `pdm generate-dataset-json-for-qa-model`. 
- Copy the folder "./data/protected/qa_json" into the other repository and train the models.
- Copy the resulting model files back into ./serialized_models/inselbert_qa_hf and run `pdm save-qa-model-to-bento` to create a BentoService for the model.


### inselbert_seq_labelling

This folder contains the model variants fine-tuned for sequence labelling, each with k-fold cross-validation.
The model takes a clinical text as input and returns annotated tokens with the corresponding labels.

#### Training
- Run `pdm finetune-sequence-labelling-model` to fine-tune the model on the training data.

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

## Evaluation

#### Evaluation of pre-trained models 

- Metric: model perplexity
    - Adapt model paths in src/further_pretraining/evaluation/exec_evaluation.py 
    - Run `pdm evaluate-pretrained-models` to evaluate the pre-trained models on the evaluation data set (10 % of data).

#### Evaluation of question answering model
- Metric: squad_v2 (including no_answer metric), with bootstrap of validation dataset
    - Make sure to have a trained model in the serialized_models folder and the constants.py file set to the correct model variants
    - run `pdm evaluate-qa-model` to evaluate the question answering model

#### Evaluation of sequence labelling model
- Metric: seqeval
    - Evaluation data is created during model training due to k-fold cross-validation
    - Analyse the generated evaluation files by running the file `src/sequence_labelling_model/analyze_cross_validation.py`
    - If needed, adapt model paths directly in the file

## Open issues

- Context information: How can we integrate context information (which fact was identified, position of anchor, position of fact within wider context) into the model?
- Performance drop upon production mode: see https://github.com/bentoml/BentoML/issues/2267
- Sharing of modifiers: What impact does sharing of modifiers have on performance?
- Ignore special tokens: Multi-label classification does not automatically ignore -100 tokens when calculating loss. 
- Merging facts: What impact does merging of facts with the same modifiers have on model performance and structuring? 
