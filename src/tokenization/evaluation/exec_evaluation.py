from transformers import AutoTokenizer

import constants


def evaluate_tokenizer():
    # Evaluation
    old_tokenizer = AutoTokenizer.from_pretrained(constants.BASE_MODEL_NAME)
    new_tokenizer = AutoTokenizer.from_pretrained(constants.PRETRAINED_MODEL_PATH)

    example = "\nCT SCHÄDEL NATIV UND MIT KM UND CT ANGIOGRAFIE DER ZERVIKALEN UND INTRAKRANIELLEN GEFÄSSE VOM 31.03.2022\n\nIndikation/Fragestellung: Unbeobachteter Sturz / Fraktur? Blutung? Stroke?\n\nBefund: Zum Vergleich die Voruntersuchung vom 20.03.2021 vorliegend.\n\nSchädel: Mittelständige Falx. Bekannte subkortikal betonte globale Hirnatrophie mit unveränderter Erweiterung der inneren Liquorräume (Seitenventrikelvorderhorn-Abstand bis max. 4,5 cm). Kein Anhalt für eine Liquorzirkulationsstörung."

    new_encoding = new_tokenizer(example)
    old_encoding = old_tokenizer(example)
    print("Token count trained tokenizer: ", len(new_encoding.tokens()))
    print("Tokens trained tokenizer: ", new_encoding.tokens())
    print("Token count base tokenizer: ", len(old_encoding.tokens()))
    print("Tokens base tokenizer: ", old_encoding.tokens())
