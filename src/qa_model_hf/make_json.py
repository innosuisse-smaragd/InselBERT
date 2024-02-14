import json

import constants
from shared.cas_loader import CASLoader
from shared.dataset_helper import DatasetHelper
from shared.schema_generator import SchemaGenerator

schema = SchemaGenerator()
loader = CASLoader(constants.ANNOTATED_REPORTS_PATH, schema)
train_dictlist, eval_test_dictlist = loader.load_CAS_convert_to_offset_dict_qa_train_test_split()
dataset_qa_train_test_valid = DatasetHelper.create_data_splits_qa(train_dictlist, eval_test_dictlist)

print(dataset_qa_train_test_valid)
train_set = dataset_qa_train_test_valid["train"]
train_set.to_json(constants.QA_DATA_OUTPUT_PATH + "/train.json", force_ascii=False)
test_set = dataset_qa_train_test_valid["test"]
test_set.to_json(constants.QA_DATA_OUTPUT_PATH + "/test.json",force_ascii=False)
validation_set = dataset_qa_train_test_valid["validation"]
validation_set.to_json(constants.QA_DATA_OUTPUT_PATH + "/validation.json",force_ascii=False)

