"""Run the six k comparisons and three chunk-setting comparisons from the assignment."""
from __future__ import annotations

import csv
from pathlib import Path

from techfix_rag import answer_question, open_vector_store

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "evaluation" / "retrieval_experiments.csv"

K_QUESTIONS = [
    ("K01", "What is the current return request window and its conditions?", "current_policy.md"),
    ("K02", "Does TF-NVR8 include a storage drive?", "products_services.csv"),
    ("K03", "What details should I give if an indoor CCTV camera is missing from the app?", "operations_faq.md"),
    ("K04", "What is the former return window?", "archived_policy.md"),
    ("K05", "Can I schedule a remote setup consultation and what does it cover?", "products_services.csv;operations_faq.md"),
    ("K06", "What should I do if there are sparks near the equipment?", "current_policy.md;operations_faq.md"),
]
CHUNK_QUESTIONS = [
    ("C01", "What is the current return request window and its conditions?", "current_policy.md"),
    ("C02", "Compare the current and previous normal appointment notice rules.", "current_policy.md;archived_policy.md"),
    ("C03", "What details should I give if an indoor CCTV camera is missing from the app?", "operations_faq.md"),
]


def main() -> None:
    rows = []
    trial_count = len(K_QUESTIONS) * 3 + len(CHUNK_QUESTIONS) * 2
    completed = 0
    db = open_vector_store(650, 100)
    for question_id, question, expected in K_QUESTIONS:
        for k in (1, 3, 5):
            completed += 1
            rows.append(_run_one("k_comparison", question_id, question, 650, 100, k, expected, db))
            print(f"Finished {completed}/{trial_count}: {question_id}, k={k}")

    for question_id, question, expected in CHUNK_QUESTIONS:
        for size, overlap in ((400, 60), (800, 120)):
            db = open_vector_store(size, overlap)
            completed += 1
            rows.append(
                _run_one("chunking", question_id, question, size, overlap, 3, expected, db)
            )
            print(f"Finished {completed}/{trial_count}: {question_id}, chunk={size}/{overlap}")

    columns = [
        "experiment", "question_id", "question", "chunk_size", "chunk_overlap", "k",
        "expected_source_files", "retrieved_sources", "relevant_evidence_found",
        "irrelevant_evidence_count", "actual_answer", "retrieved_excerpts",
        "answer_correctness", "notes",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved trial evidence to {OUTPUT}")
    print("Manually complete answer_correctness and notes after reviewing each answer and excerpt.")


def _run_one(experiment, question_id, question, chunk_size, overlap, k, expected, db):
    try:
        result = answer_question(question, k=k, vector_db=db)
        expected_sources = [item.strip() for item in expected.split(";") if item.strip()]
        retrieved_sources = set(result.sources)
        relevant = all(item in retrieved_sources for item in expected_sources)
        irrelevant = len(retrieved_sources - set(expected_sources))
        notes = "Review answer and retrieved excerpts; automatic source coverage is not an accuracy judgment."
        answer = result.answer.replace("\n", " ")
    except Exception as exc:
        result = None
        relevant = False
        irrelevant = ""
        notes = f"ERROR: {exc}"
        answer = ""
        retrieved_sources = set()
    return {
        "experiment": experiment,
        "question_id": question_id,
        "question": question,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "k": k,
        "expected_source_files": expected,
        "retrieved_sources": ";".join(sorted(retrieved_sources)),
        "relevant_evidence_found": "Yes" if relevant else "No",
        "irrelevant_evidence_count": irrelevant,
        "actual_answer": answer,
        "retrieved_excerpts": " || ".join(
            f"{item['source_file']}: {item['excerpt'][:250]}"
            for item in (result.evidence if result else [])
        ),
        "answer_correctness": "REVIEW",
        "notes": notes,
    }


if __name__ == "__main__":
    main()
