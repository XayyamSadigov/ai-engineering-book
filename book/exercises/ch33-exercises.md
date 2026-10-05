# Exercises — Chapter 33 — Fine-Tuning for Engineers

Solutions: `../solutions/ch33-solutions.md`


### Knowledge questions

**K1.** Explain why a weekly-changing HR policy should not be fine-tuned into a model, using the "compiler, not database" model and the properties of auditability and revocability.

**K2.** For a 2048 × 8192 weight matrix, compute the LoRA trainable parameters at ranks 8 and 32, and the ratio to full fine-tuning. Then name two memory costs LoRA does not reduce.

**K3.** Why is loss computed only on assistant tokens in SFT? What goes wrong in a hand-written trainer that forgets to mask?

**K4.** A team reports a fine-tuned classifier at 0.91 accuracy and 0.58 macro-F1 on 40 classes. What does the gap tell you, and which single additional table would you ask for?

**K5.** Describe what DPO removes relative to the classic RLHF pipeline, and give two reasons an application team would still not run DPO themselves.

**K6.** Distinguish exact deduplication, near deduplication, and entity-grouped splitting. Give one leak each one catches that the other two miss.

### Engineering questions

**E1.** Northwind wants to fine-tune the same small base for three tasks: ticket classification, invoice field extraction, and reply drafting in house style. Design the versioning and serving scheme: adapters or merged weights, how many model ids, what the registry records, and how a prompt change on one task is gated.

**E2.** The strong model's labels are the training targets, but agents override 9% of them. Design the filtering and sampling policy for distillation: which rows become training targets, how overrides are used, what fraction gets human review, and how rare categories are protected.

**E3.** Design the monitoring and retraining policy for the cascade: which three signals, what thresholds and durations, what forces an immediate retrain versus a scheduled one, and what evidence the retrain must produce before replacing the model id.

**E4.** A hosted provider announces the base model behind your adapter will be retired in 90 days. Write the migration plan, including what the registry must already contain for this to be routine.

### Practical exercises

**P1.** Make the near-dedup embedder part of the dataset's provenance. Extend `build_dataset` so the data card records which embedder ran (`tfidf`, or the `embedding_space` of an `aie_core` client passed through `embed_fn_from_client`) and the threshold. Then write a test with `FakeEmbeddings(vocabulary=...)` that shows a reworded pair the TF-IDF fallback keeps and the vocabulary embedder removes, and asserts that two builds with different embedders produce cards that say so.

**P2.** Add a stratification check to the data card: for each label, the share in train, val, and test, and a warning list of labels whose test share is zero or whose train share is below a floor. Add a test with a corpus that triggers the warning.

**P3.** Implement a memorization probe: given the training JSONL and a prediction function, prompt with the first half of each user turn for a sample of rows and report how many completions reproduce the second half above a similarity threshold. Test it with a fake model that memorizes.

**P4.** Write a second `FineTuneProvider` adapter for a different REST shape of your choosing (different endpoint names and status vocabulary), tested with `httpx.MockTransport`, and show that `run_fine_tune` and its tests do not change.

### Debugging exercises

**D1.** A fine-tuned classifier scored macro-F1 0.88 on the holdout. In production, agent overrides are three times higher than that implies. The data card shows the split strategy was `entity` with a 0.8/0.1/0.1 fraction and dedup counts of zero for both exact and near. Inspecting traces shows many tickets from an outage storm in the last month of the export. Diagnose the cause and name the two fields in the card that should have warned you.

**D2.** After a retrain, the escalation rate fell from 14% to 3% overnight and the override rate rose from 7% to 12%. Training loss was lower than the previous run and the validation macro-F1 was slightly higher. The run used four epochs instead of two. Explain what happened and which metric in the ship rule should have blocked the release.

**D3.** A fine-tuned extraction model performs perfectly on the holdout and fails on 30% of tickets from a newly added intake channel. The failing tickets differ from the training ones only in that they lack a boilerplate footer the old channel appended. Name the failure, the check that would have flagged it before training, and the slice that would have caught it in evaluation.
