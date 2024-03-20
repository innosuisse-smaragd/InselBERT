import streamlit as st
import itertools

import constants
from transformers import pipeline


class InferenceFact():
    def __init__(self, id, span, start, end, score, alternatives):
        self.id = id
        self.span = span
        self.start = start
        self.end = end
        self.score = score
        self.alternatives = alternatives
        self.extracted_tokens = {}
        self.merged_tokens = {}
        self.extracted_entities = {}

    id: str
    span: str
    start: int
    end: int
    score: float
    alternatives: str
    extracted_tokens: object = {}
    merged_tokens: object = {}
    extracted_entities: object = {}


def words_overlap(slice1, slice2):
    if slice1[0] < slice2[0]:  # slice1 is leftmost
        return slice2[0] < slice1[1]  # slice2 ends before slice1 starts
    else:
        return slice1[0] < slice2[1]

st.set_page_config(
    page_title="Smaragd QA-based fact extraction",
    page_icon=":gem:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:daniel.reichenpfader@bfh.ch',
        'Report a bug': "mailto:daniel.reichenpfader@bfh.ch",
        'About': """Showcase of fact-based information extraction from radiology reports, based on InselBERT.
                 This app is part of the Smaragd project.
                 """
    }
)

st.title(":gem: Smaragd QA-based fact extraction")
model_checkpoint_qa = "./serialized_models/inselbert_qa_hf/"
model_checkpoint_seq_labelling = "./serialized_models/inselbert_seq_labelling/"
question_answerer = pipeline("question-answering", model=model_checkpoint_qa, handle_impossible_answer=True)
sequence_labeller = pipeline("ner", model=model_checkpoint_seq_labelling)

st.write("""
Copy your dummy report text, extract all available facts or choose one or multiple facts on the left. 
""")


facts_list = constants.FILTERED_FACT_DEFINITIONS

toggle = st.sidebar.toggle('Extract all facts', True)

if not toggle:
    fact_names = st.sidebar.multiselect(
        'Select Fact(s)',
        facts_list
    )

report_text = st.text_area("Dummy report text", value= constants.EXAMPLE_TOKENIZED_REPORT, height=250)

facts = []
alternatives = []
# Step 1: Extract facts from the report by extractive question answering

if st.button("Extract facts"):
    print(report_text)
    with st.spinner('Extracting facts...'):
        if toggle:
            fact_definitions = constants.FILTERED_FACT_DEFINITIONS
        else:
            fact_definitions = fact_names
        for fact_names in fact_definitions:
            answers = question_answerer(question=fact_names, context=report_text, top_k=3, handle_impossible_answer=True)
            print("Raw answers: ",answers)

            if answers[0]['answer'] == "":
                continue
            else:
                pairs_of_results = itertools.permutations(answers, 2)
                for pair in pairs_of_results:

                    if words_overlap((pair[0]['start'], pair[0]['end']), (pair[1]['start'], pair[1]['end'])):
                        if pair[0]['score'] > pair[1]['score']:
                            if pair[1] in answers:
                                answers.remove(pair[1])
                                alternatives.append(pair[1])
                        else:
                            if pair[0] in answers:
                                answers.remove(pair[0])
                                alternatives.append(pair[0])
                for key, answer in enumerate(answers):
                    if answer['score'] < 0.2:  # TODO: maybe make this a parameter
                        alternatives.append(answer)
                    else:
                        facts.append(InferenceFact(id=fact_names, span=answer['answer'], score=answer['score'], start=answer['start'], end=answer['end'], alternatives=str(alternatives)))
            # Step 2: Labelling of anchors and modifiers

        for fact in facts:
            print("Fact: ", fact.id)
            extracted_tokens = sequence_labeller(inputs=fact.span)
            fact.extracted_tokens = extracted_tokens

        # TODO: Maybe sort out tags with low scores

            # Merge subword-tokens
            results = []
            prev_tok = None
            prev_end = 0
            prev_start = 0
            for tag in extracted_tokens:
                if tag['word'].startswith("##") and prev_tok is not None:
                    prev_tok += tag['word'][2:]
                    prev_end = tag['end']
                else:
                    if prev_tok is not None:
                        results.append({"word": prev_tok, "tag": prev_tag, "start": prev_start, "end": prev_end})
                    prev_tok = tag['word']
                    prev_end = tag['end']
                    prev_tag = tag['entity']
                    prev_start = tag['start']
            if prev_tok is not None:
                results.append({"word": prev_tok, "tag": prev_tag, "start": prev_start, "end": prev_end})
            fact.merged_tokens = results

            # Merge B- and I- tags to one tag
            merged_tags = []
            prev_word = None
            prev_end = 0
            prev_start = 0
            for tag in fact.merged_tokens:
                if tag['tag'].startswith("I-") and prev_word is not None:
                    prev_word += " " + tag['word']
                    prev_end = tag['end']
                else:
                    if prev_word is not None:
                        merged_tags.append({"word": prev_word, "tag": prev_tag[2:], "start": prev_start, "end": prev_end})
                    prev_word = tag['word']
                    prev_tag = tag['tag']
                    prev_end = tag['end']
                    prev_start = tag['start']
            if prev_word is not None:
                merged_tags.append({"word": prev_word, "tag": prev_tag[2:], "start": prev_start, "end": prev_end})
            fact.extracted_entities = merged_tags

        if not facts:
            st.warning("No facts found.")
        else:
            st.markdown("### Extracted fact(s):")
            for key, fact in enumerate(facts):
                with st.expander(fact.id):

                    st.markdown("## *" + fact.span + "*")
                    st.metric(label="Score", value=str(fact.score))
                    entity_table_header = ("#### Extracted entities: \n | Entity | Value | \n | --- | --- | \n")

                    for extracted_entity in fact.extracted_entities:
                        entity_table_header += ("|" + extracted_entity['tag'] + "|" + extracted_entity['word'] + "| \n")
                    st.markdown(entity_table_header)

            st.markdown("#### Alternatives:")
            with st.expander("Show alternatives"):
                st.write(alternatives)

