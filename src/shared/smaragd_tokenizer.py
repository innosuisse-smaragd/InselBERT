#based on https://github.com/innosuisse-smaragd/smaragd-shared-python/blob/main/smaragd_shared_python/pre_processing/spacy_tokenizer.py
# hard copy due to smaragd-shared currently being private and therefore not accessible during deployment

import spacy
from spacy.lang.char_classes import (ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, CURRENCY, LIST_CURRENCY,
                                     LIST_ELLIPSES, LIST_HYPHENS, LIST_ICONS, LIST_PUNCT, LIST_QUOTES, PUNCT, UNITS,
                                     _quotes)
from spacy.lang.de import German
from spacy.tokenizer import ORTH, Tokenizer

import constants


class SmaragdTokenizer:
    """
    Note that this tokenizer was not built on top of the German Spacy tokenizer.
    Instead, appropriate rules were copies from the langauge-independent and language-dependent Spacy rules.
    This has the advantage that all rules are at one place (in this class).
    Language-independent Spacy rules:
    https://github.com/explosion/spaCy/blob/eab929361d0ecff914f6d7c9973ea552c77f678a/spacy/lang/tokenizer_exceptions.py#L60
    German Spacy punctuation rules:
    https://github.com/explosion/spaCy/blob/master/spacy/lang/punctuation.py
    German Spacy rules:
    https://github.com/explosion/spaCy/blob/master/spacy/lang/de/tokenizer_exceptions.py
    """

    def _get_spacy_tokenizer(self, nlp):
        special_cases = {}

        with open("./shared/ID-abbreviation-list.txt", 'r', encoding="utf8") as id_abbreviation_file:
            for line in id_abbreviation_file:
                line_without_trailing_whitespaces = line.rstrip()
                special_cases[line_without_trailing_whitespaces] = [{ORTH: line_without_trailing_whitespaces}]

        EXTENDED_UNITS = UNITS + "|Gy|gy|Hz|hz|Bq|bq|Sv|sv|mol|A|mAs"

        prefixes = (
                ["§", "%", "=", "—", "–", r"\+(?![0-9])"]
                + LIST_PUNCT
                + LIST_ELLIPSES
                + LIST_QUOTES
                + LIST_CURRENCY
                + LIST_ICONS
                + ["``"]
                # Custom rules follow
                + LIST_HYPHENS  # -12/2022 -> - 12 / 2022 | -Mammografie -> - Mammografie
                + [r"\/"]
        )
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)

        suffixes = (
                LIST_PUNCT
                + LIST_ELLIPSES
                + LIST_QUOTES
                + LIST_ICONS
                + ["'s", "'S", "’s", "’S", "—", "–"]
                + [
                    r"(?<=[0-9])\+",
                    r"(?<=°[FfCcKk])\.",
                    r"(?<=[0-9])(?:{c})".format(c=CURRENCY),
                    r"(?<=[0-9])(?:{u})".format(u=EXTENDED_UNITS),
                    r"(?<=[0-9{al}{e}{p}(?:{q})])\.".format(
                        al=ALPHA_LOWER, e=r"%²\-\+", q=CONCAT_QUOTES, p=PUNCT
                    ),
                    r"(?<=[{au}][{au}])\.".format(au=ALPHA_UPPER),
                ]
                + ["''", "/"]
                # Custom rules follow
                + [
                    r"(?<=[0-9])(?:{u})".format(u="J|j|J.|j.|Y|y|Y.|y.|Uhr|uhr|H|h|H.|h."),  # 49J -> 49, J
                    r"(?<=[0-9])(?:{u})".format(u="\\."),  # BI-RADS 2. -> BI - RADS 2 . | 11.12.2023. -> 11.12.2023 .
                ]
        )
        suffix_regex = spacy.util.compile_suffix_regex(suffixes)

        infix_chars = LIST_PUNCT + LIST_HYPHENS + LIST_QUOTES + [r"\/"] + [r"\+"]
        # Because of decimal numbers , is no infix
        infix_chars = [char for char in infix_chars if char != ","]
        infixes = (
                LIST_ELLIPSES
                + LIST_ICONS
                + [
                    r"(?<=[{al}])\.(?=[{au}])".format(al=ALPHA_LOWER, au=ALPHA_UPPER),
                    r"(?<=[{a}])[,!?](?=[{a}])".format(a=ALPHA),
                    r"(?<=[{a}])[:<>=](?=[{a}])".format(a=ALPHA),
                    r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                    r"(?<=[0-9{a}])\/(?=[0-9{a}])".format(a=ALPHA),
                    r"(?<=[{a}])([{q}\)\]\(\[])(?=[{a}])".format(a=ALPHA, q=_quotes),
                    r"(?<=[{a}])--(?=[{a}])".format(a=ALPHA),
                    r"(?<=[0-9])-(?=[0-9])",
                ]
                # Custom rules follow
                + [
                    r"(?<=[0-9]){u}(?=[0-9])".format(u="X|x")  # 30x30x30 -> 30 x 30 x 30
                ]
                + infix_chars
        )
        infix_regex = spacy.util.compile_infix_regex(infixes)

        return Tokenizer(nlp.vocab,
                         rules=special_cases,
                         prefix_search=prefix_regex.search,
                         suffix_search=suffix_regex.search,
                         infix_finditer=infix_regex.finditer,
                         url_match=None)

    def __init__(self):
        super().__init__()
        self.nlp = German()
        self.nlp.tokenizer = self._get_spacy_tokenizer(self.nlp)

    def tokenize(self, text: str) -> [(int, int)]:
        """
        Returns a list of bi-tuples where each tuple has the start index of the token as first element
        and the end index as second element.
        """
        doc = self.nlp(text)

        with doc.retokenize() as retokenizer:
            for token in doc:
                if token.is_space:
                    whitespace_characters = [char for char in token.text]
                    heads = []
                    number_whitespace_characters = len(whitespace_characters)
                    for index, whitespace_character in enumerate(whitespace_characters):
                        if number_whitespace_characters - 1 - index == 0:
                            heads.append(token.head)
                        else:
                            heads.append((token, number_whitespace_characters - 1 - index))

                    retokenizer.split(token, whitespace_characters, heads=heads)

        tokens = [(token.idx, token.idx + len(token.text)) for token in doc if " " not in token.text]

        #added by DR:
        tokenized_tokens_text = [text[token[0]:token[1]] for token in tokens]
        return tokenized_tokens_text

if __name__ == "__main__":
    tokenizer = SmaragdTokenizer()
    text = "MAMMOGRAFIE BEIDSEITS IN ZWEI EBENEN VOM 22.06.2021. Met. Adeno-Ca, unkl. Primarius. (CT-Befund vom 10.6.21 mit Vd. a. i.e.L. DD HCC, DD CCC."
    tokenized_text = tokenizer.tokenize(text)
    print(tokenized_text)