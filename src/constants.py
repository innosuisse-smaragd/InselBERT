REPORTS_CSV_FILE_PATH = "./data/protected/radiology-reports.csv"
REPORTS_CSV_FILE_COLUMN_NAME="text"
ANNOTATED_REPORTS_PATH = "./data/protected/annotated-reports"
FACT_SCHEMA_PATH="./data/test/fact_schema_v41.html"

BASE_MODEL_NAME ="GerMedBERT/medbert-512"
F_A_EXTRACTION_MODEL_NAME = "inselbert_extract_f_a"
M_EXTRACTION_MODEL_NAME = "inselbert_extract_m"
QA_MODEL_NAME = "inselbert_qa"
QA_HF_MODEL_NAME = "inselbert_qa_hf"
SEQ_LABELLING_MODEL_NAME = "inselbert_seq_labelling"

PRETRAINED_MODEL_PATH = "./serialized_models/inselbert/"
F_A_EXTRACTION_MODEL_PATH = "./serialized_models/" + F_A_EXTRACTION_MODEL_NAME + "/"
M_EXTRACTION_MODEL_PATH = "./serialized_models/" + M_EXTRACTION_MODEL_NAME + "/"
QA_HF_MODEL_PATH = "./serialized_models/" + QA_HF_MODEL_NAME + "/"
SEQ_LABELLING_MODEL_PATH = "./serialized_models/" + SEQ_LABELLING_MODEL_NAME + "/"

BENTO_NAME = "inselbertfactextractor"
LABEL_ALL_TOKENS = True

FACTS = "FACTS"
ANCHORS = "ANCHORS"
MODIFIERS = "MODIFIERS"
BINARIES = "BINARIES"


