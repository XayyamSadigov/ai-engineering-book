# Chapter 26: threat models and attack corpus

Small, dependency-free modules that support Chapter 26 (Threat Modeling and Prompt Injection) and feed
Chapter 27's guardrail red team.

| File | Public names | Purpose |
|---|---|---|
| `attack_corpus.py` | `Variant`, `HarmfulEffect`, `SensitiveDocument`, `AdversarialDocument`, `sensitive_documents`, `adversarial_documents`, `find_canary_leaks`, `extract_urls`, `extract_image_urls`, `off_allowlist_urls`, `decode_base64_blocks`, `write_corpus` | One adversarial carrier per injection technique, canary-stamped sensitive records, and effect detectors |
| `threat_model.py` | `Trust`, `Stride`, `Level`, `Asset`, `Principal`, `Boundary`, `EntryPoint`, `Threat`, `ThreatModel`, `render_threat_table`, `render_markdown`, `northwind_rag_model`, `northwind_agent_model` | Threat models as reviewable data, with validation, risk ordering, and the control list Chapter 27 implements |
| `test_ch26.py` | | Offline tests that assert effects, not wording |

No environment variables, no network, no API keys. All destinations use reserved `.example` and
`.invalid` domains.

```bash
# from the repository root
.venv/bin/python -m pytest book/projects/examples/ch26 -q
# render the worked threat models
cd book/projects/examples/ch26 && python -c "import threat_model as t; print(t.render_markdown(t.northwind_agent_model()))"
# write the attack corpus to a folder for your own pipeline tests
python -c "from pathlib import Path; import attack_corpus as ac; print(ac.write_corpus(Path('corpus')))"
```

Chapter 27's `guardrails` package locates this directory through `GUARDRAILS_CH26_DIR` (default
`../examples/ch26` relative to the package).
