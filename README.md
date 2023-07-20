# LLM-based, two-layer clinical fact extraction based on the approach by Steinkamp et al.

## Description of serialized models

### medbert_512

This model serves as the basis for further development. The folder contains the git-repo of the model itself. More information [here](https://huggingface.co/GerMedBERT/medbert-512).

### medbert_512_pretrained

This folder contains the result of the "training" of a tokenizer based on Insel radiology reports. The tokenizer is first initialized based on the medbert_512 model's tokenizer and then completely "trained" on the insel corpus based on the WordPiece algorithmn. The output of a tokenizer "training" comprises the following files:

- special_tokens_map.json
- tokenizer_config.json
- tokenizer.json
- vocab.txt

Furthermore, this folder contains the result of the further-pretraining of medbert_512 on the Insel corpus based on Masked Language Modelling (MLM) and the trained tokenizer.

### medbert_512_facts

This folder contains the fine-tuned model which is composed of the medbert_512_pretrained encoder body and two multi-label classification heads. The heads predict for each token to which facts (0..n) it might belong and wether the token corresponds to an anchor entity, respectively. This architecture corresponds to the method proposed by Steinkamp et al. Once trained, this model takes a clinical text as input and returns annotations for each token (facts and anchors).

### medbert_512_refined

This folder contains the second, fine-tuned model which is composed of the same medbert_512_pretrained encoder body of medbert_512_facts. Additionally, three classification heads are added. Two binary classification heads predict wether a token is the beginning or end of a fact span, respectively. A third, multi-class classification head predicts which modifier (0..1) a token is part of. Once trained, this model takes a fact candidate, which corresponds to the validated output of the former model, X tokens before and after the fact candidate as well as the information which fact was identified (see open issues).

## Open issues

- Context information: How can we integrate context information (which fact was identified, position of anchor, position of fact within wider context) into the model?
- Fact candidate calculation: Where and how are fact candidates calculated? Is this part of one of the two models?
