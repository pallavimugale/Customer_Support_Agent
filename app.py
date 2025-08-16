import streamlit as st
import faiss
import pandas as pd
from src.helper import semantic_similarity, call_llm
import config

# ---- Functions ----

def display_response(query, df, index):
    # call semantic similarity
    distances, indices = semantic_similarity(query, index, config.EMBEDDING_MODEL)
    top_similar_instructions = df.iloc[indices[0]].reset_index(drop=True)
    top_similar_instructions['distance'] = distances[0]

    st.write("Following responses will be generated:\n"
             "1. Urgency of the query on a scale of 1-5.\n"
             "2. Categorize the query into sales, product, operations etc.\n"
             "3. Generated Response from LLM.")
    st.write("## Response from LLM below (wait a few seconds)")

    llm_response = call_llm(query, top_similar_instructions['response'].tolist())
    st.write(llm_response)

    return llm_response, top_similar_instructions

def handle_query_input(idx, df, index):
    """Reusable block for handling one query input and response."""
    query = st.text_input("User Query:", key=f"query_{idx}")
    if st.button("Get response", key=f"btn_{idx}"):
        if not query:
            st.error("Please enter a query.")
        else:
            llm_response, top_similar_instructions = display_response(query, df, index)
            st.session_state.queries.append(query)
            st.session_state.responses.append(llm_response)
            st.session_state.similar.append(top_similar_instructions)
            st.rerun()  # Refresh UI immediately

# ---- Main App ----

st.title("IntelliServe - AI-Driven Customer Helpdesk")

# initialize session state for storing multiple queries/responses
if "queries" not in st.session_state:
    st.session_state.queries = []
if "responses" not in st.session_state:
    st.session_state.responses = []
if "similar" not in st.session_state:
    st.session_state.similar = []

# load vector DB and dataset
index = faiss.read_index(config.VECTOR_INDEX_FILE_PATH)
df = pd.read_csv(config.CUSTOMER_SUPPORT_TRAINNING_DATASET)

# Display history
for i, (q, r) in enumerate(zip(st.session_state.queries, st.session_state.responses)):
    st.markdown(f"### Query {i+1}: {q}")
    st.markdown(f"**Response:** {r}")
    
    # Feedback section for each query
    with st.expander("Feedback / Regenerate"):
        feedback = st.text_area("Provide feedback:", key=f"feedback_{i}")
        if st.button("Accept Response", key=f"accept_{i}"):
            st.success("Response has been submitted.")
        if st.button("Regenerate Response", key=f"regen_{i}"):
            if feedback:
                # generate new query for LLM
                new_query = (
                    f"Regenerate third point of this response: {r}.\n"
                    "You must only regenerate the third point according to the feedback below. "
                    "Do not change the 1st and 2nd points but always include them in the final output.\n"
                    f"Feedback: {feedback}"
                )
                st.write("## New Response from LLM (wait a few seconds)")
                new_llm_response = call_llm(new_query, st.session_state.similar[i]['response'].tolist())
                st.write(new_llm_response)
                st.session_state.responses[i] = new_llm_response
                st.rerun()
            else:
                st.error("Please provide feedback to regenerate.")

# Always render one fresh query input at the bottom
next_idx = len(st.session_state.queries)
handle_query_input(next_idx, df, index)
