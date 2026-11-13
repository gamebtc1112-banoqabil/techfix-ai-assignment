# TechFix AI Assignment

TechFix Supply & Service is a fictional HVAC/CCTV equipment online store with a technical support desk. All company, product, policy, and operational facts are synthetic classroom data. This separate project folder does not use the earlier Northstar files.

## What is included

- Five source documents totaling 1,500–2,500 words: company overview, product/service catalog, current policy, archived policy, and operations FAQ.
- Module 7 style TextLoader and CSVLoader ingestion, with source filename and policy version/status/date metadata on chunks.
- Recursive chunking, MiniLM embeddings, persistent Chroma storage, and configurable retrieval k in the Streamlit sidebar.
- A Groq-backed answer function with document-only grounding, current-versus-archived policy selection, abstention for missing facts, visible citations, and an expandable evidence panel.
- A 20-case test plan and an in-app evaluation dashboard that exports answers, citations, excerpts, and review fields.
- A 24-trial retrieval experiment plan for k=1/3/5 and two alternate chunk settings.

## Setup on Windows

Open a terminal in this folder and run:

    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    Copy-Item .env.example .env

Open .env in VS Code and replace the placeholder with your own GROQ_API_KEY. Keep .env private. Do not paste the key into source files, screenshots, chat, or a public repository. The default model can be changed with GROQ_MODEL.

Start the chatbot:

    streamlit run streamlit_app.py

Streamlit prints a local address, usually http://localhost:8501. The first startup downloads the embedding model and builds the local Chroma index; later runs reuse it. After editing or adding company documents, click Rebuild knowledge index or restart the app. The index is excluded from source control and ZIP submission; the code recreates it from the documents. If a policy changes but a persistent index is not refreshed, old chunks may still be retrieved. Rebuild and rerun the current-versus-archived tests.

Optional terminal chat:

    python techfix_rag.py

## Evaluation procedure

1. Before running, review evaluation/test_cases.csv. It contains 20 questions across factual lookup, paraphrases, two-source questions, current/archived policy, missing information, and prompt injection. Expected answers and sources are specified before actual results.
2. Run the Streamlit evaluation dashboard and download the results CSV. Check each answer against the retrieved excerpts, then replace REVIEW with PASS or FAIL and provide a specific reason for every failure. A matching source filename alone does not prove the answer is correct.
3. From a terminal, run python run_retrieval_experiments.py to record six questions at k=1, 3, and 5 plus three questions with chunk settings 400/60 and 800/120. This makes 24 answer calls. Review the saved excerpts, complete the answer correctness and notes fields, and discuss evidence coverage, irrelevant chunks, and answer quality.
4. Discuss at least two failure cases, the diagnosis, and the correction or limitation. Do not label unrun trials as passed.

## Demo checklist

Show a normal product fact, a question needing two documents, a current-versus-archived policy comparison, a missing detail, a prompt-injection test, and one evaluation result/failure investigation. Explain the loader, chunk metadata, persistent index, retrieval k, grounded answer, and how changing a source policy triggers re-indexing. Record a 3–5 minute video and put its accessible link in this README before submission.

## Submission contents

Submit this project folder as a ZIP, the completed 20-test CSV, a report of at most three pages, and the video link. Keep the ZIP under 5 MB. Exclude .venv, .rag_store, model downloads, .env, and API credentials. The CSV source catalog and Markdown docs are synthetic; no private customer information is included.
