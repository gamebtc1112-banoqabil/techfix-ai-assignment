from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd
import streamlit as st


def _show_table(frame, *args, hide_index=False, **kwargs):
    """Render a small evaluation table without Streamlit's PyArrow conversion."""
    html = frame.to_html(index=not hide_index, escape=True, border=0)
    st.markdown(
        '<div style="overflow-x:auto;max-height:600px">' + html + '</div>',
        unsafe_allow_html=True,
    )

from techfix_rag import (
    ABSTENTION,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DOCS_DIR,
    AnswerResult,
    answer_question,
    open_vector_store,
)

st.set_page_config(page_title="TechFix AI Support", page_icon="🛠️", layout="wide")
st.title("TechFix AI — HVAC & CCTV Support")
st.caption("Fictional classroom support assistant. Answers use only the supplied synthetic company documents.")

with st.sidebar:
    st.subheader("Retrieval settings")
    retrieval_k = st.slider("Retrieved chunks (k)", min_value=1, max_value=10, value=4)
    st.caption("Each chunk keeps its source filename and policy version/status.")
    if st.button("Rebuild knowledge index", help="Use after changing or adding a company document."):
        st.cache_resource.clear()
        st.session_state["force_rebuild"] = True


@st.cache_resource(show_spinner=False)
def get_index(force: bool = False):
    return open_vector_store(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, force_rebuild=force)


def show_answer(result: AnswerResult) -> None:
    st.markdown(result.answer)
    if result.sources:
        st.caption("Sources: " + ", ".join(result.sources))
    with st.expander("Retrieved evidence and source excerpts"):
        for number, evidence in enumerate(result.evidence, start=1):
            st.markdown(
                f"**{number}. {evidence['source_file']}** · "
                f"Policy {evidence['policy_version']} · {evidence['policy_status']}"
            )
            st.write(evidence["excerpt"])
            st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    with st.spinner("Loading documents and knowledge index..."):
        vector_db = get_index(bool(st.session_state.pop("force_rebuild", False)))
    st.success(f"Knowledge base ready · {len(vector_db.get()['ids'])} chunks from {len(list(DOCS_DIR.iterdir()))} documents", icon="✅")
except Exception as exc:
    vector_db = None
    st.error(f"Knowledge base could not be loaded: {exc}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["answer"])
        if message["role"] == "assistant":
            if message.get("sources"):
                st.caption("Sources: " + ", ".join(message["sources"]))
            with st.expander("Retrieved evidence and source excerpts"):
                for evidence in message.get("evidence", []):
                    st.markdown(
                        f"**{evidence['source_file']}** · Policy "
                        f"{evidence['policy_version']} · {evidence['policy_status']}"
                    )
                    st.write(evidence["excerpt"])

if question := st.chat_input("Ask about HVAC/CCTV products, support, or policy"):
    st.session_state.messages.append({"role": "user", "answer": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if vector_db is None:
            st.error("The document index is not available. Review the message above, then restart after fixing the setup.")
        else:
            try:
                with st.spinner("Searching company documents..."):
                    result = answer_question(question, k=retrieval_k, vector_db=vector_db)
                show_answer(result)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "answer": result.answer,
                        "sources": result.sources,
                        "evidence": result.evidence,
                    }
                )
            except Exception as exc:
                st.error(f"Could not complete the answer: {exc}")

with st.expander("Evaluation dashboard — 20 fixed assignment tests"):
    st.caption("Run after the key and document index are ready. Review evidence and label results manually; source overlap alone is not proof of correctness.")
    test_path = Path(__file__).parent / "evaluation" / "test_cases.csv"
    try:
        tests = pd.read_csv(test_path).fillna("")
    except Exception as exc:
        tests = pd.DataFrame()
        st.error(f"Could not read the test plan: {exc}")

    _show_table(
        tests[["test_id", "category", "question", "expected_source_files", "expected_answer"]]
        if not tests.empty else tests,
        use_container_width=True,
        hide_index=True,
    )
    if st.button("Run 20 tests", type="primary", disabled=vector_db is None or tests.empty):
        output_rows = []
        progress = st.progress(0, text="Starting evaluation...")
        for index, row in tests.iterrows():
            try:
                result = answer_question(str(row["question"]), k=retrieval_k, vector_db=vector_db)
                actual_sources = ", ".join(result.sources)
                expected_sources = [
                    source.strip()
                    for source in str(row["expected_source_files"]).split(";")
                    if source.strip()
                ]
                citation_present = all(source in result.sources for source in expected_sources)
                output_rows.append(
                    {
                        **row.to_dict(),
                        "actual_answer": result.answer,
                        "actual_citations": actual_sources,
                        "expected_source_retrieved": "Yes" if citation_present else "No",
                        "manual_pass_fail": "REVIEW",
                        "failure_reason": "Judge grounded answer, version selection, abstention, and prompt-injection resistance against expected evidence.",
                        "retrieved_excerpts": " || ".join(
                            f"{item['source_file']}: {item['excerpt'][:200]}"
                            for item in result.evidence
                        ),
                    }
                )
            except Exception as exc:
                output_rows.append(
                    {
                        **row.to_dict(),
                        "actual_answer": "",
                        "actual_citations": "",
                        "expected_source_retrieved": "Error",
                        "manual_pass_fail": "FAIL",
                        "failure_reason": str(exc),
                        "retrieved_excerpts": "",
                    }
                )
            progress.progress((index + 1) / len(tests), text=f"Completed {index + 1} of {len(tests)}")
        progress.empty()
        result_df = pd.DataFrame(output_rows)
        st.session_state["techfix_eval_results"] = result_df

    if "techfix_eval_results" in st.session_state:
        result_df = st.session_state["techfix_eval_results"]
        _show_table(result_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download results CSV",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="techfix_20_test_results.csv",
            mime="text/csv",
        )
        st.info("REVIEW means a person must check the actual answer and evidence before changing it to PASS or FAIL.")

with st.expander("Safety and scope"):
    st.write(
        "This classroom assistant is not an emergency service, electrician, or repair authority. "
        "For smoke, sparks, burning odor, exposed wiring, water near powered equipment, or repeated "
        "breaker trips, stop using the equipment if safe and contact a qualified professional."
    )
