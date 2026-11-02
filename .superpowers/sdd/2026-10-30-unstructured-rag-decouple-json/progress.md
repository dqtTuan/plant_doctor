# SDD ledger — plan: docs/superpowers/plans/2026-10-30-unstructured-rag-decouple-json.md
Pre-flight: Task 1 migrates 30 diseases to markdown and deletes diseases.json. Task 2 implements query_expansion and decouples knowledge.py. Task 3 adapts build_index and training modules. Task 4 updates pipeline and UI for open semantic search. Task 5 runs full verification and updates docs.
Task 1: complete (commit c11d66c, converted 31 diseases to cam_nang_benh_cay_trong.md and deleted diseases.json)
Task 2: complete (commit d89bc52, tests: pytest test_knowledge & test_query_expansion -> 13 passed)
Task 3: complete (commit 8bf057a, tests: pytest test_unstructured_rag, test_rag, test_training_interfaces -> 4 passed)
Task 4: complete (commit 78f43fe, tests: pytest test_pipeline -> 6 passed)
Task 5: complete (commit 6379fdb, tests: pytest full suite -> 29 passed)
