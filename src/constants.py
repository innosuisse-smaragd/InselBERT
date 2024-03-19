REPORTS_CSV_FILE_PATH = "./data/protected/radiology-reports.csv"
REPORTS_CSV_FILE_COLUMN_NAME = "text"
ANNOTATED_REPORTS_PATH = "./data/protected/annotated-reports"
FACT_SCHEMA_PATH = "./data/test/fact_schema_v41.html"
QA_DATA_OUTPUT_PATH = "./data/protected/qa_json"

BASE_MODEL_NAME = "GerMedBERT/medbert-512"
F_A_EXTRACTION_MODEL_NAME = "inselbert_extract_f_a"
M_EXTRACTION_MODEL_NAME = "inselbert_extract_m"
QA_MODEL_NAME = "inselbert_qa"
QA_HF_MODEL_NAME = "inselbert_qa_hf"
SEQ_LABELLING_MODEL_NAME = "inselbert_seq_labelling"

PRETRAINED_MODEL_PATH = "./serialized_models/inselbert/"
F_A_EXTRACTION_MODEL_PATH = "./serialized_models/" + F_A_EXTRACTION_MODEL_NAME + "/"
M_EXTRACTION_MODEL_PATH = "./serialized_models/" + M_EXTRACTION_MODEL_NAME + "/"
QA_MODEL_PATH = "./serialized_models/" + QA_MODEL_NAME + "/"
QA_HF_MODEL_PATH = "./serialized_models/" + QA_HF_MODEL_NAME + "/"
SEQ_LABELLING_MODEL_PATH = "./serialized_models/" + SEQ_LABELLING_MODEL_NAME + "/"

TOKENIZER_ABBREVIATIONS_FILE = "./data/ID-abbreviation-list.txt"

BENTO_NAME = "inselbertfactextractor"
LABEL_ALL_TOKENS = True

FACTS = "FACTS"
ANCHORS = "ANCHORS"
MODIFIERS = "MODIFIERS"
BINARIES = "BINARIES"

# to avoid download of smaragd-shared-python during deployment
FACT_DEFINITIONS = {"Herdläsion beschrieben",
                    "Verkalkung beschrieben",
                    "BI-RADS Klassifizierung erwähnt",
                    "Parenchymdichte beschrieben",
                    "Kutis beschrieben",
                    "ACR Klassifizierung erwähnt",
                    "Subkutis beschrieben",
                    "Lymphknoten beschrieben",
                    "Mammillenregion beschrieben",
                    "Empfehlung für weitere Untersuchung",
                    "Architekturstörung beschrieben",
                    "Fremdmaterial beschrieben",
                    "Asymmetrie beschrieben",
                    "Zusätzlicher Befund",
                    "Vergleich mit Voruntersuchung erwähnt",
                    "Zuweisungsinformation erwähnt",
                    "Intrammäre Lymphknoten beschrieben",
                    "Bildgebendes Verfahren durchgeführt",
                    "Untersuchung hat Limitierung",
                    "Klinischer Befund / Anamnese / Diagnose"}

FILTER_ENTITIES = True

ENTITIES_TO_IGNORE = {
    "Klinischer Befund / Anamnese / Diagnose", # 2
    "Information / Befund / Anamnese / Diagnose", # anchor for fact above
    "Untersuchung hat Limitierung", # 5
    "Limitierung", # corresponding anchor to fact above
    "Bildgebendes Verfahren durchgeführt", # 7
    "Intrammäre Lymphknoten beschrieben", # 8
    "Intramammärer Lymphknoten", # anchor for fact above
    # "Zuweisungsinformation erwähnt", # 10
    # "Vergleich mit Voruntersuchung erwähnt", # 10
    "Asymmetrie_progredient",
    "Abteilung/Arzt",
    "Rand",
    "Asymmetrie_fokal",
    "Asymmetrie_Rand",
    "Position_Clip",
    "Asymmetrie_global",
    "Zeit/Bedingung",
    "Status nach",
    # "Verkalkung_Rand" # 10
}

EXAMPLE_TOKENIZED_REPORT = "MAMMOGRAFIE BEIDSEITS IN ZWEI EBENEN VOM 22.06.2021&#10;&#10;Fragestellung/Indikation&#10;Met. Adeno-Ca, unkl. Primarius. (CT-Befund vom 10.6.21 mit Vd. a. i.e.L. DD HCC, DD CCC.&#10;Positive Familienanamnese für Brustkrebs (Tante väterlicherseits).&#10;Malignität/Auffälligkeiten?&#10;&#10;Klinische Untersuchung und Ultraschall:&#10;Durchführung in der gynäkologischen Senologie dieser Klinik (Bericht siehe dort).&#10;&#10;Befund&#10;Zum Vergleich liegt die Voruntersuchung vom 28.09.2020 vor.&#10;&#10;Mammografie beidseits MLO und CC:&#10;Kutis und Subkutis unauffällig.&#10;Mittelfleckiges, teilweise involutiertes Drüsenparenchym beidseits.&#10;Kein malignomsuspekter Herdbefund. Keine suspekte Mikrokalkgruppe.&#10;&#10;Beurteilung&#10;Mammographisch kein Anhalt für Malignität beidseits.&#10;&#10;ACR-Typ b beidseits.&#10;BIRADS 1 beidseits.&#10;&#10;Falls auch der Ultraschallbefund der Brust unauffällig sein sollte, wäre eine weitere Befundabsicherung mittels MR Mammographie zu erwägen."
