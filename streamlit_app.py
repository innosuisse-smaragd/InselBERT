import streamlit as st
import itertools

from shared.schema_generator import SchemaGenerator
from transformers import pipeline


st.set_page_config(page_title="Smaragd QA-based fact extraction", page_icon=":gem:")
st.title("Smaragd QA-based fact extraction")
model_checkpoint = "./serialized_models/inselbert_qa_hf/"
question_answerer = pipeline("question-answering", model=model_checkpoint, handle_impossible_answer=True)


st.write("""
Copy your dummy report text, choose a fact on the left or extract all available facts. 
""")

#toggle = st.sidebar.toggle('Extract all facts')

schema = SchemaGenerator()
facts = schema.id2tag_facts.values()

#if not toggle:
fact_name = st.sidebar.selectbox(
    'Select Fact',
    facts
)

report_text = st.text_area("Dummy report text", height=250)


def words_overlap(slice1, slice2):
    if slice1[0] < slice2[0]:  # slice1 is leftmost
        return slice2[0] < slice1[1]  # slice2 ends before slice1 starts
    else:
        return slice1[0] < slice2[1]

alternatives = []

if st.button("Extract facts"):
    answers = question_answerer(question=fact_name, context=report_text, top_k=3)
    if answers[0]['answer'] == "":
        st.write("The report does not contain the specified fact.")
    else:
        pairs_of_results = itertools.permutations(answers, 2)
        for pair in pairs_of_results:
            print(pair)
            if words_overlap((pair[0]['start'], pair[0]['end']), (pair[1]['start'], pair[1]['end'])):
                if pair[0]['score'] > pair[1]['score']:

                    if pair[1] in answers:
                        answers.remove(pair[1])
                        alternatives.append(pair[1])
                else:

                    if pair[0] in answers:
                        answers.remove(pair[0])
                        alternatives.append(pair[0])

        st.markdown("### Extracted fact(s):")
        for key, answer in enumerate(answers):


            if answer['score'] < 0.4:
                alternatives.append(answer)
            else:

                st.write(answer['answer'])
                st.metric(label="Score", value=answer['score'])
                st.write("")

        st.markdown("### Alternatives:")
        st.write(alternatives)


