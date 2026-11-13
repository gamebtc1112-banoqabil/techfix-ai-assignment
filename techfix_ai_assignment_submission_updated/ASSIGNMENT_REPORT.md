# TechFix AI - Assignment Report

**Student:** MUHAMMAD AFZAL #   518550  
**Course / section:** GENERATIVE AI

**Demo video (3–5 minutes):** [Watch TechFix AI demo](https://drive.google.com/file/d/19L3nY1U-insy3m48DG2RqrcgIdRolvL9/view?usp=sharing)

**Drive project ZIP:** [Open TechFix AI project ZIP](https://drive.google.com/file/d/1oCMBx_XA-Z1M177j0-jbbwXPXbQJQwlg/view?usp=sharing)

**GitHub project:** [View TechFix AI repository](https://github.com/gamebtc1112-banoqabil/techfix-ai-assignment)

## Problem and solution

TechFix AI is a fictional HVAC/CCTV equipment store support assistant for technicians and customers. It answers questions about synthetic product listings, safe first-line support, opening hours, and customer rules. The RAG pipeline loads five company documents, splits them into overlapping chunks, creates embeddings with sentence-transformers/all-MiniLM-L6-v2, and stores them in Chroma. At query time, the system retrieves evidence, sends it with the question to the LLM, cites source files, shows excerpts, and abstains when the documents lack an answer.

## Policy and safety handling

Policy chunks retain version, status, and effective date. Version 2.0 is current from 2026-11-01; version 1.0 is archived through 2026-10-31. Current questions use the current rule, historical questions use the archive, and comparisons state both versions. The current policy includes two exceptions. Retrieved document instructions are treated as untrusted data. Safety guidance is limited to stopping use if safe and contacting a qualified professional.

## Evaluation results

**20-question test set:** Manual review records 19 PASS and 1 FAIL. T01 remains PASS because operations_faq.md supports the service details and current_policy.md supports the rescheduling rule; the products_services.csv coverage flag is incomplete, but the cited evidence supports the answer. I01 is FAIL: the response gives the current 14-day return rule but cites archived_policy.md (version 1.0), so its source does not support the current rule.

**Retrieval k comparison (1, 3, 5):** Across six questions, all expected source files were retrieved in 3/6 cases at k=1, 4/6 at k=3, and 6/6 at k=5. Irrelevant source excerpts totalled 1, 5, and 6 respectively. Manually reviewing answer correctness across all 24 retrieval trials resulted in 22 PASS and 2 FAIL.

**Chunk settings (400/60 versus 800/120):** Both settings retrieved all expected sources for all three questions and each returned one irrelevant excerpt. All six chunk-setting answers passed manual review. Results were similar; this small sample does not show a clear winner.

**Most useful observation:** Increasing k improved expected-source coverage, especially on multi-document questions, but also brought in more unrelated evidence. Changing chunk size did not improve coverage. Prompt-injection text appeared in K01 excerpts at k=3 and k=5; the answers did not follow its instruction.

**Failure case 1:** K02 at k=1 answered correctly from operations_faq.md but missed expected products_services.csv. This is a source-coverage failure, not an answer-accuracy failure; k=3 retrieved both expected sources.

**Failure case 2 and unresolved limitation:** K05 at k=1 and k=3 added an unsupported “phone call” detail and missed expected products_services.csv, although operations_faq.md supported the main service description. Both are marked FAIL. At k=5 the answer used the supported general term “call” and retrieved both expected sources, but two irrelevant sources remained. The unresolved limitation is extra retrieval noise at higher k; relevance filtering or reranking could be explored later.

## Limitations and reproducibility

The business and documents are fictional. Retrieval source coverage does not by itself prove answer correctness. The Groq API key is held locally in .env and must not be included in a repository or ZIP. Chroma is generated from the five supplied documents and excluded from submission. After changing a source policy, rebuild the index and repeat relevant current/archived tests to avoid stale evidence. This is a classroom prototype and needs further evaluation before real-world use.
