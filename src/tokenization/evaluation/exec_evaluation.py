from transformers import AutoTokenizer

import constants

# Evaluation
old_tokenizer = AutoTokenizer.from_pretrained(constants.BASE_MODEL_PATH)
new_tokenizer = AutoTokenizer.from_pretrained(constants.PRETRAINED_MODEL_PATH)

example = "Das ist der erste Befund"

new_encoding = new_tokenizer(example)
old_encoding = old_tokenizer(example)
print(len(new_encoding.tokens()))
print(new_encoding.tokens())
print(len(old_encoding.tokens()))
print(old_encoding.tokens())