import streamlit as st
import itertools

from shared.schema_generator import SchemaGenerator
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


st.set_page_config(page_title="Smaragd QA-based fact extraction", page_icon=":gem:")
st.title("Smaragd QA-based fact extraction")
model_checkpoint_qa = "./serialized_models/inselbert_qa_hf/"
model_checkpoint_seq_labelling = "./serialized_models/inselbert_seq_labelling/"
question_answerer = pipeline("question-answering", model=model_checkpoint_qa, handle_impossible_answer=True)
sequence_labeller = pipeline("ner", model=model_checkpoint_seq_labelling)

st.write("""
Copy your dummy report text, choose a fact on the left or extract all available facts. 
""")

schema = SchemaGenerator()
facts_list = schema.id2tag_facts.values()

#if not toggle:
fact_name = st.sidebar.selectbox(
    'Select Fact',
    facts_list
)

report_text = st.text_area("Dummy report text", height=250)

facts = []
alternatives = []
# Step 1: Extract facts from the report by extractive question answering

if st.button("Extract facts"):
    answers = question_answerer(question=fact_name, context=report_text, top_k=3, handle_impossible_answer=True)
    print("Raw answers: ",answers)

    if answers[0]['answer'] == "":
        st.write("The report does not contain the specified fact.")
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
                facts.append(InferenceFact(id=fact_name, span=answer['answer'], score=answer['score'], start=answer['start'], end=answer['end'], alternatives=str(alternatives)))
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

    #toggle = st.sidebar.toggle('Extract all facts'

st.markdown("### Extracted fact(s):")
for key, fact in enumerate(facts):
    st.write("Fact: " + fact.id)
    st.write("Span: " + fact.span)
    st.metric(label="Score", value=str(fact.score))
    st.write("")
    st.markdown("#### Extracted entities:")
    for extracted_entity in fact.extracted_entities:
        st.write(extracted_entity['tag'] + ": " + extracted_entity['word'])


st.markdown("### Alternatives:")
st.write(alternatives)


