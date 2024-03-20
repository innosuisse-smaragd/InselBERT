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
FILTERED_FACT_DEFINITIONS = {"Herdläsion beschrieben",
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
                             #  "Vergleich mit Voruntersuchung erwähnt",
                             #  "Zuweisungsinformation erwähnt",
                             #  "Intrammäre Lymphknoten beschrieben",
                             # "Bildgebendes Verfahren durchgeführt",
                             # "Untersuchung hat Limitierung",
                             # "Klinischer Befund / Anamnese / Diagnose"
                             }

AVAILABLE_MODIFIERS = {
    # Modifiers:
    "Lateralität": 1576,
    "Zustand": 661,
    "Lokalisierung": 445,
    "Negation": 373,
    "Dynamik": 270,
    "Grösse": 224,
    "Verkalkung_Verteilung": 220,
    "Dignität": 203,
    "Position_Uhrzeit": 196,
    "Parenchymdichte_Transparenz": 180,
    "Position_Mamillenabstand": 173,
    "Herdläsion_Rand": 158,
    "Unsicherheit": 153,
    "Position_Quadrant": 133,
    "Projektionsebene": 128,
    "Verkalkung_Dignität": 121,
    "Begleitmerkmale": 94,
    "Verdächtige Morphologie": 83,
    "Anzahl": 80,
    "Verdachtsdiagnose": 63,
    "Parenchymdichte_Form": 54,
    "Diagnoseverfahren": 48,  # Also an anchor^
    "Herdläsion_Form": 44,
    "Zeit / Datum": 31,
    "Herdläsion_Dichte": 26,
    "Typisch benigne Veränderung": 23,
    "Parenchymdichte_Rand": 19,
    "Untersuchung": 19,
    "Parenchymdichte_Dignität": 13,
    "Indikation": 12,
    "Verkalkung_Rand": 10,
    "Status nach": 9,
    "Zeit/Bedingung": 5,
    "Asymmetrie_global": 3,
    "Position_Clip": 3,
    "Asymmetrie_Rand": 3,
    "Asymmetrie_fokal": 2,
    "Rand": 2,
    "Abteilung/Arzt": 1,
    "Asymmetrie_progredient": 1
}
AVAILABLE_FACTS = {
    "Herdläsion beschrieben": 479,
    "Verkalkung beschrieben": 455,
    "BI-RADS Klassifizierung erwähnt": 346,
    "Parenchymdichte beschrieben": 280,
    "Kutis beschrieben": 252,
    "ACR Klassifizierung erwähnt": 213,
    "Subkutis beschrieben": 186,
    "Lymphknoten beschrieben": 160,
    "Mammillenregion beschrieben": 110,
    "Empfehlung für weitere Untersuchung": 100,
    "Architekturstörung beschrieben": 87,
    "Fremdmaterial beschrieben": 41,
    "Asymmetrie beschrieben": 34,
    "Zusätzlicher Befund": 31,
    "Vergleich mit Voruntersuchung erwähnt": 10,
    "Zuweisungsinformation erwähnt": 10,
    "Intrammäre Lymphknoten beschrieben": 8,
    "Bildgebendes Verfahren durchgeführt": 7,
    "Untersuchung hat Limitierung": 5,
    "Klinischer Befund / Anamnese / Diagnose": 2,
}
AVAILABLE_ANCHORS = {
    # Anchors:
    # 'Indikation / Fragestellung': 1,
    'Lymphknoten': 2,
    # 'Diagnoseverfahren': 3,
    # 'Solitärer erweiterter Gang': 4,
    'Fremdmaterial': 5,
    'Kutis': 6,
    'Parenchym': 7,
    # 'Therapie': 8,
    'Architekturstörung': 9,
    'ACR Klassifizierung': 10,
    'Verkalkung': 11,
    'BI-RADS Klassifizierung': 12,
    'Befund': 13,
    'Herdläsion': 14,
    'Empfehlung': 15,
    'Asymmetrie': 16,
    'Subkutis': 17,
    'Modalität': 18,
    'Mammillenregion': 19,
    'Voruntersuchung': 2000,
    'Zuweiser*in': 2001,
    'Intramammärer Lymphknoten': 2002,
    'Limitierung': 2003,
    'Information / Befund / Anamnese / Diagnose': 2004
}

FILTER_ENTITIES = True

ENTITIES_TO_IGNORE = {
    # Fact and anchor tags with n < 20 excluded below (see data/protected/total_counts.json)
    "Klinischer Befund / Anamnese / Diagnose",  # 2
    "Information / Befund / Anamnese / Diagnose",  # anchor for fact above
    "Untersuchung hat Limitierung",  # 5
    "Limitierung",  # corresponding anchor to fact above
    "Bildgebendes Verfahren durchgeführt",  # 7
    "Modalität",  # problem: also a modifier tag, but this is not used
    "Intrammäre Lymphknoten beschrieben",  # 8
    "Intramammärer Lymphknoten",  # anchor for fact above
    "Zuweisungsinformation erwähnt",  # 10
    "Zuweiser*in",  # anchor for fact above
    "Vergleich mit Voruntersuchung erwähnt",  # 10
    "Voruntersuchung",  # anchor for fact above
    # Modifier tags with n < 20 excluded below
    "Asymmetrie_progredient",
    "Abteilung/Arzt",
    "Rand",
    "Asymmetrie_fokal",
    "Asymmetrie_Rand",
    "Position_Clip",
    "Asymmetrie_global",
    "Zeit/Bedingung",
    "Status nach",
    "Verkalkung_Rand",
    "Indikation",
    "Parenchymdichte_Dignität",
    "Untersuchung",
    "Parenchymdichte_Rand"
}

EXAMPLE_TOKENIZED_REPORT = "MAMMOGRAFIE BEIDSEITS IN ZWEI EBENEN VOM 22.06.2021&#10;&#10;Fragestellung/Indikation&#10;Met. Adeno-Ca, unkl. Primarius. (CT-Befund vom 10.6.21 mit Vd. a. i.e.L. DD HCC, DD CCC.&#10;Positive Familienanamnese für Brustkrebs (Tante väterlicherseits).&#10;Malignität/Auffälligkeiten?&#10;&#10;Klinische Untersuchung und Ultraschall:&#10;Durchführung in der gynäkologischen Senologie dieser Klinik (Bericht siehe dort).&#10;&#10;Befund&#10;Zum Vergleich liegt die Voruntersuchung vom 28.09.2020 vor.&#10;&#10;Mammografie beidseits MLO und CC:&#10;Kutis und Subkutis unauffällig.&#10;Mittelfleckiges, teilweise involutiertes Drüsenparenchym beidseits.&#10;Kein malignomsuspekter Herdbefund. Keine suspekte Mikrokalkgruppe.&#10;&#10;Beurteilung&#10;Mammographisch kein Anhalt für Malignität beidseits.&#10;&#10;ACR-Typ b beidseits.&#10;BIRADS 1 beidseits.&#10;&#10;Falls auch der Ultraschallbefund der Brust unauffällig sein sollte, wäre eine weitere Befundabsicherung mittels MR Mammographie zu erwägen."
