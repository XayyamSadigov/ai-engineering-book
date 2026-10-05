# References and Further Reading

This book is self-contained. Every concept it relies on is explained in the chapters, and every project
runs from the code in the repository. The references below are optional: they point to the primary
papers behind the techniques, the specifications the code follows, and the documentation of the tools
it uses. Read them when you want the original argument, the full experimental evidence, or the exact
normative wording.

Two cautions apply. First, specifications and tool documentation move fast. Protocols such as the Model
Context Protocol, the OpenTelemetry generative-AI semantic conventions, and the OWASP list for LLM
applications are revised regularly, so always check the current version rather than relying on what
the book or a paper describes. Second, a paper's reported numbers come from its own models, data, and
setup. Treat them as evidence that an idea can work, then measure it on your data with the evaluation
methods of Part VII.

Paper links point to the arXiv abstract page where one exists. Entries without an arXiv version give a
DOI or the publisher's page.

---

## Part I: AI Engineering Foundations (Chapters 1 to 3)

**Transformer and tokenization**

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and
  Polosukhin, I. (2017). *Attention Is All You Need.* NeurIPS. https://arxiv.org/abs/1706.03762
- Sennrich, R., Haddow, B., and Birch, A. (2016). *Neural Machine Translation of Rare Words with
  Subword Units.* ACL. The paper that brought byte-pair encoding to neural language models.
  https://arxiv.org/abs/1508.07909
- Kudo, T., and Richardson, J. (2018). *SentencePiece: A Simple and Language Independent Subword
  Tokenizer and Detokenizer for Neural Text Processing.* EMNLP (demo). https://arxiv.org/abs/1808.06226

**Sampling, attention variants, and constrained generation**

- Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. (2020). *The Curious Case of Neural Text
  Degeneration.* ICLR. Introduces nucleus (top-p) sampling. https://arxiv.org/abs/1904.09751
- Shazeer, N. (2019). *Fast Transformer Decoding: One Write-Head is All You Need.* Multi-query
  attention. https://arxiv.org/abs/1911.02150
- Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., and Sanghai, S. (2023). *GQA:
  Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP.
  https://arxiv.org/abs/2305.13245
- Willard, B. T., and Louf, R. (2023). *Efficient Guided Generation for Large Language Models.*
  Grammar- and schema-constrained decoding. https://arxiv.org/abs/2307.09702

**Long context**

- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. (2024).
  *Lost in the Middle: How Language Models Use Long Contexts.* TACL. https://arxiv.org/abs/2307.03172

**Reliability patterns used by the gateway**

- Brooker, M. (2015). *Exponential Backoff and Jitter.* AWS Architecture Blog.
  https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
- Amazon Builders' Library. *Timeouts, retries, and backoff with jitter.*
  https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/

## Part II: LLM Application Development (Chapters 4 to 7)

- Brown, T. B., et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS. The paper that made
  in-context (few-shot) learning a practical technique. https://arxiv.org/abs/2005.14165
- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., and Zhou, D.
  (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.
  https://arxiv.org/abs/2201.11903
- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. (2023).
  *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR.
  https://arxiv.org/abs/2203.11171
- Khattab, O., et al. (2023). *DSPy: Compiling Declarative Language Model Calls into Self-Improving
  Pipelines.* Prompts as optimizable programs; also relevant to Chapter 23.
  https://arxiv.org/abs/2310.03714
- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). *On Calibration of Modern Neural
  Networks.* ICML. Background for confidence thresholds, cascades, and abstention.
  https://arxiv.org/abs/1706.04599

## Part III: Embeddings and Retrieval (Chapters 8 and 9)

**Embedding models**

- Reimers, N., and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks.* EMNLP. https://arxiv.org/abs/1908.10084
- Kusupati, A., et al. (2022). *Matryoshka Representation Learning.* NeurIPS.
  https://arxiv.org/abs/2205.13147
- Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. (2023). *MTEB: Massive Text Embedding
  Benchmark.* EACL. Useful for shortlisting models, never a substitute for evaluation on your data.
  https://arxiv.org/abs/2210.07316

**Approximate nearest neighbor search**

- Malkov, Y. A., and Yashunin, D. A. (2018). *Efficient and Robust Approximate Nearest Neighbor Search
  Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI (published 2020).
  https://arxiv.org/abs/1603.09320
- Jégou, H., Douze, M., and Schmid, C. (2011). *Product Quantization for Nearest Neighbor Search.*
  IEEE TPAMI 33(1). https://doi.org/10.1109/TPAMI.2010.57
- Johnson, J., Douze, M., and Jégou, H. (2017). *Billion-scale Similarity Search with GPUs.* The FAISS
  paper, covering IVF and PQ at scale. https://arxiv.org/abs/1702.08734

## Part IV: Production RAG (Chapters 10 to 15)

**Retrieval-augmented generation and dense retrieval**

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
  NeurIPS. https://arxiv.org/abs/2005.11401
- Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W. (2020). *Dense
  Passage Retrieval for Open-Domain Question Answering.* EMNLP. https://arxiv.org/abs/2004.04906
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and Gurevych, I. (2021). *BEIR: A Heterogeneous
  Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets and Benchmarks.
  Shows why BM25 remains a strong baseline. https://arxiv.org/abs/2104.08663

**Lexical retrieval, fusion, and reranking**

- Robertson, S., and Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
  Foundations and Trends in Information Retrieval 3(4). The standard account of BM25, which originated
  in the Okapi system at City University London in the 1990s. https://doi.org/10.1561/1500000019
- Cormack, G. V., Clarke, C. L. A., and Büttcher, S. (2009). *Reciprocal Rank Fusion Outperforms
  Condorcet and Individual Rank Learning Methods.* SIGIR. https://doi.org/10.1145/1571941.1572114
- Nogueira, R., and Cho, K. (2019). *Passage Re-ranking with BERT.* The cross-encoder reranking
  baseline. https://arxiv.org/abs/1901.04085
- Gao, L., Ma, X., Lin, J., and Callan, J. (2023). *Precise Zero-Shot Dense Retrieval without Relevance
  Labels.* ACL. Introduces HyDE. https://arxiv.org/abs/2212.10496
- Anthropic (2024). *Introducing Contextual Retrieval.* An engineering write-up of chunk
  contextualization before indexing. https://www.anthropic.com/news/contextual-retrieval

**Deduplication**

- Broder, A. Z. (1997). *On the Resemblance and Containment of Documents.* Proceedings of Compression
  and Complexity of Sequences. The origin of MinHash for near-duplicate detection.

**Retrieval and RAG evaluation**

- Järvelin, K., and Kekäläinen, J. (2002). *Cumulated Gain-Based Evaluation of IR Techniques.* ACM
  Transactions on Information Systems 20(4). The definition of (n)DCG.
  https://doi.org/10.1145/582415.582418
- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2024). *RAGAS: Automated Evaluation of
  Retrieval Augmented Generation.* EACL (demo). https://arxiv.org/abs/2309.15217
- Min, S., et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form
  Text Generation.* EMNLP. Claim-level verification. https://arxiv.org/abs/2305.14251

## Part V: Tools and Workflows (Chapters 16 to 18)

- Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., and
  Scialom, T. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools.* NeurIPS.
  https://arxiv.org/abs/2302.04761
- Hardy, N. (1988). *The Confused Deputy (or why capabilities might have been invented).* ACM SIGOPS
  Operating Systems Review 22(4). The original confused-deputy paper, directly relevant to tool and MCP
  security. https://doi.org/10.1145/54289.871709
- Saltzer, J. H., and Schroeder, M. D. (1975). *The Protection of Information in Computer Systems.*
  Proceedings of the IEEE 63(9). The source of least privilege and fail-safe defaults.
  https://doi.org/10.1109/PROC.1975.9939

## Part VI: Agents (Chapters 19 to 23)

**Agent loops and architectures**

- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. (2023). *ReAct:
  Synergizing Reasoning and Acting in Language Models.* ICLR. https://arxiv.org/abs/2210.03629
- Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., and Yao, S. (2023). *Reflexion:
  Language Agents with Verbal Reinforcement Learning.* NeurIPS. https://arxiv.org/abs/2303.11366
- Madaan, A., et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback.* NeurIPS. A
  generate, critique, revise loop; read alongside Chapter 20's caution about self-critique without new
  evidence. https://arxiv.org/abs/2303.17651

**Memory**

- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., and Bernstein, M. S. (2023).
  *Generative Agents: Interactive Simulacra of Human Behavior.* UIST. Memory retrieval by recency,
  importance, and relevance, plus reflection. https://arxiv.org/abs/2304.03442
- Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., and Gonzalez, J. E. (2023).
  *MemGPT: Towards LLMs as Operating Systems.* Tiered memory with explicit paging.
  https://arxiv.org/abs/2310.08560

**Multi-agent systems**

- Wu, Q., et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.*
  https://arxiv.org/abs/2308.08155
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. (2023). *Improving Factuality and
  Reasoning in Language Models through Multiagent Debate.* https://arxiv.org/abs/2305.14325

## Part VII: Evaluation (Chapters 24 and 25)

- Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS Datasets
  and Benchmarks. Documents position, verbosity, and self-enhancement biases.
  https://arxiv.org/abs/2306.05685
- Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., and Zhu, C. (2023). *G-Eval: NLG Evaluation using GPT-4
  with Better Human Alignment.* EMNLP. https://arxiv.org/abs/2303.16634
- Wang, P., et al. (2023). *Large Language Models are not Fair Evaluators.* Position bias in pairwise
  judging and mitigations. https://arxiv.org/abs/2305.17926
- Liang, P., et al. (2022). *Holistic Evaluation of Language Models (HELM).* Multi-metric,
  scenario-based evaluation design. https://arxiv.org/abs/2211.09110
- Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales.* Educational and Psychological
  Measurement 20(1). The original kappa statistic. https://doi.org/10.1177/001316446002000104
- Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife.* The Annals of Statistics 7(1).
  https://doi.org/10.1214/aos/1176344552
- Kohavi, R., Tang, D., and Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical
  Guide to A/B Testing.* Cambridge University Press. Background for online evaluation and canaries.

## Part VIII: Security and Guardrails (Chapters 26 and 27)

- Perez, F., and Ribeiro, I. (2022). *Ignore Previous Prompt: Attack Techniques For Language Models.*
  NeurIPS ML Safety Workshop. https://arxiv.org/abs/2211.09527
- Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., and Fritz, M. (2023). *Not What You've
  Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.*
  AISec. https://arxiv.org/abs/2302.12173
- Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J. Z., and Fredrikson, M. (2023). *Universal and
  Transferable Adversarial Attacks on Aligned Language Models.* Automated jailbreak suffixes.
  https://arxiv.org/abs/2307.15043
- Hines, K., Lopez, G., Hall, M., Zarfati, F., Zunger, Y., and Kiciman, E. (2024). *Defending Against
  Indirect Prompt Injection Attacks With Spotlighting.* Delimiting and datamarking untrusted input.
  https://arxiv.org/abs/2403.14720
- Wallace, E., Xiao, K., Leike, R., Weng, L., Heidecke, J., and Beutel, A. (2024). *The Instruction
  Hierarchy: Training LLMs to Prioritize Privileged Instructions.* https://arxiv.org/abs/2404.13208
- Debenedetti, E., Zhang, J., Balunović, M., Beurer-Kellner, L., Fischer, M., and Tramèr, F. (2024).
  *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.*
  NeurIPS Datasets and Benchmarks. https://arxiv.org/abs/2406.13352

## Part IX: Production AI Engineering (Chapters 28 to 34)

**Reliability, scalability, and architecture**

- Beyer, B., Jones, C., Petoff, J., and Murphy, N. R. (eds.) (2016). *Site Reliability Engineering.*
  O'Reilly. Free online with its companion books; the reference for SLOs and error budgets.
  https://sre.google/books/
- Dean, J., and Barroso, L. A. (2013). *The Tail at Scale.* Communications of the ACM 56(2).
  https://doi.org/10.1145/2408776.2408794
- Little, J. D. C. (1961). *A Proof for the Queuing Formula: L = λW.* Operations Research 9(3).
  https://doi.org/10.1287/opre.9.3.383
- Nygard, M. T. (2018). *Release It! Design and Deploy Production-Ready Software*, 2nd ed. Pragmatic
  Bookshelf. Circuit breakers, bulkheads, timeouts, and stability anti-patterns.
- Fowler, M. *CircuitBreaker.* https://martinfowler.com/bliki/CircuitBreaker.html
- Kleppmann, M. (2017). *Designing Data-Intensive Applications.* O'Reilly. Queues, idempotence,
  delivery guarantees, and event logs.

**Engineering practices**

- Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS. Still the
  best short account of why the model is the small part of the system.
- Evans, E. (2003). *Domain-Driven Design.* Addison-Wesley. The anti-corruption layer pattern used for
  provider abstraction.
- Nygard, M. (2011). *Documenting Architecture Decisions.* The original proposal for ADRs.
  https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions

**Fine-tuning and preference optimization (Chapter 33)**

- Houlsby, N., et al. (2019). *Parameter-Efficient Transfer Learning for NLP.* ICML. Adapter modules.
  https://arxiv.org/abs/1902.00751
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2022).
  *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR. https://arxiv.org/abs/2106.09685
- Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of
  Quantized LLMs.* NeurIPS. https://arxiv.org/abs/2305.14314
- Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., and Amodei, D. (2017). *Deep
  Reinforcement Learning from Human Preferences.* NeurIPS. https://arxiv.org/abs/1706.03741
- Ouyang, L., et al. (2022). *Training Language Models to Follow Instructions with Human Feedback.*
  NeurIPS. The InstructGPT paper: SFT followed by RLHF. https://arxiv.org/abs/2203.02155
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. (2023). *Direct
  Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS.
  https://arxiv.org/abs/2305.18290
- Zhou, C., et al. (2023). *LIMA: Less Is More for Alignment.* NeurIPS. Evidence that a small,
  high-quality SFT set can go far. https://arxiv.org/abs/2305.11206
- Hinton, G., Vinyals, O., and Dean, J. (2015). *Distilling the Knowledge in a Neural Network.*
  https://arxiv.org/abs/1503.02531
- Kirkpatrick, J., et al. (2017). *Overcoming Catastrophic Forgetting in Neural Networks.* PNAS.
  https://arxiv.org/abs/1612.00796

**Inference and serving (Chapter 34)**

- Pope, R., et al. (2023). *Efficiently Scaling Transformer Inference.* MLSys. Prefill versus decode,
  memory bandwidth, and KV cache costs. https://arxiv.org/abs/2211.05102
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. (2022). *FlashAttention: Fast and
  Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS. https://arxiv.org/abs/2205.14135
- Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., and Chun, B.-G. (2022). *Orca: A Distributed Serving
  System for Transformer-Based Generative Models.* OSDI. Iteration-level (continuous) batching.
  https://www.usenix.org/conference/osdi22/presentation/yu
- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica,
  I. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP.
  The vLLM paper. https://arxiv.org/abs/2309.06180
- Zheng, L., et al. (2024). *SGLang: Efficient Execution of Structured Language Model Programs.*
  NeurIPS. Prefix sharing with RadixAttention. https://arxiv.org/abs/2312.07104
- Zhong, Y., et al. (2024). *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large
  Language Model Serving.* OSDI. A source of the goodput framing. https://arxiv.org/abs/2401.09670
- Leviathan, Y., Kalman, M., and Matias, Y. (2023). *Fast Inference from Transformers via Speculative
  Decoding.* ICML. https://arxiv.org/abs/2211.17192
- Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., and Jumper, J. (2023). *Accelerating
  Large Language Model Decoding with Speculative Sampling.* https://arxiv.org/abs/2302.01318
- Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., and Dao, T. (2024). *Medusa: Simple LLM
  Inference Acceleration Framework with Multiple Decoding Heads.* ICML.
  https://arxiv.org/abs/2401.10774
- Li, Y., Wei, F., Zhang, C., and Zhang, H. (2024). *EAGLE: Speculative Sampling Requires Rethinking
  Feature Uncertainty.* ICML. https://arxiv.org/abs/2401.15077
- Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L. (2022). *LLM.int8(): 8-bit Matrix
  Multiplication for Transformers at Scale.* NeurIPS. https://arxiv.org/abs/2208.07339
- Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. (2023). *GPTQ: Accurate Post-Training
  Quantization for Generative Pre-trained Transformers.* ICLR. https://arxiv.org/abs/2210.17323
- Lin, J., et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression and
  Acceleration.* MLSys. https://arxiv.org/abs/2306.00978

## Parts X and XI: System Design and Advanced Patterns (Chapters 35 to 38)

- Khattab, O., and Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via
  Contextualized Late Interaction over BERT.* SIGIR. https://arxiv.org/abs/2004.12832
- Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., and Zaharia, M. (2022). *ColBERTv2:
  Effective and Efficient Retrieval via Lightweight Late Interaction.* NAACL.
  https://arxiv.org/abs/2112.01488
- Edge, D., et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization.* The GraphRAG paper. https://arxiv.org/abs/2404.16130
- Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. (2024). *Self-RAG: Learning to Retrieve,
  Generate, and Critique through Self-Reflection.* ICLR. An adaptive-retrieval approach related to
  agentic RAG. https://arxiv.org/abs/2310.11511
- Yu, T., et al. (2018). *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain
  Semantic Parsing and Text-to-SQL Task.* EMNLP. The benchmark that made execution-based evaluation of
  text-to-SQL standard; background for Chapter 36's analytics case. https://arxiv.org/abs/1809.08887
- Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. (2024).
  *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR. The standard benchmark
  framing for coding agents. https://arxiv.org/abs/2310.06770
- Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., and Press, O. (2024).
  *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.* NeurIPS. Harness and
  interface design for coding agents. https://arxiv.org/abs/2405.15793

---

## Specifications and standards

Check the current version of each before relying on details; several are revised more than once a
year.

- **Model Context Protocol (MCP).** Specification, concepts, and SDKs. https://modelcontextprotocol.io
- **JSON-RPC 2.0 Specification.** The message format MCP builds on. https://www.jsonrpc.org/specification
- **OpenTelemetry Semantic Conventions for Generative AI.** Standard attribute names for model calls,
  token usage, and agent operations. https://opentelemetry.io/docs/specs/semconv/gen-ai/
- **W3C Trace Context.** How trace identifiers propagate across services.
  https://www.w3.org/TR/trace-context/
- **OWASP Top 10 for Large Language Model Applications.**
  https://owasp.org/www-project-top-10-for-large-language-model-applications/
- **NIST AI Risk Management Framework (AI RMF 1.0, NIST AI 100-1, 2023).**
  https://www.nist.gov/itl/ai-risk-management-framework
- **NIST AI 600-1, Generative Artificial Intelligence Profile (2024).** The generative-AI companion to
  the AI RMF. https://doi.org/10.6028/NIST.AI.600-1
- **MITRE ATLAS.** A knowledge base of adversary tactics and techniques against AI systems.
  https://atlas.mitre.org
- **JSON Schema.** The schema language for tool parameters and structured outputs.
  https://json-schema.org
- **Server-sent events, HTML Living Standard.**
  https://html.spec.whatwg.org/multipage/server-sent-events.html
- **RFC 6585, Additional HTTP Status Codes.** Defines 429 Too Many Requests.
  https://www.rfc-editor.org/rfc/rfc6585
- **RFC 9110, HTTP Semantics.** Includes the Retry-After header and the meaning of status codes.
  https://www.rfc-editor.org/rfc/rfc9110
- **RFC 7519, JSON Web Token (JWT).** https://www.rfc-editor.org/rfc/rfc7519

## Documentation of tools used in the book

**Core stack**

- Python packaging with uv: https://docs.astral.sh/uv/
- Pydantic (v2): https://docs.pydantic.dev
- FastAPI: https://fastapi.tiangolo.com
- HTTPX: https://www.python-httpx.org
- pytest: https://docs.pytest.org
- Hypothesis (property-based testing): https://hypothesis.readthedocs.io
- OpenTelemetry for Python: https://opentelemetry.io/docs/languages/python/
- tiktoken (tokenizer used by `aie_core.llm.tokens` when installed): https://github.com/openai/tiktoken
- Jinja (templating): https://jinja.palletsprojects.com
- Docker Compose: https://docs.docker.com/compose/
- Redis: https://redis.io/docs/

**Data and retrieval**

- pgvector: https://github.com/pgvector/pgvector
- PostgreSQL full-text search (`tsvector`, `tsquery`, ranking):
  https://www.postgresql.org/docs/current/textsearch.html
- Sentence Transformers (local embedding and cross-encoder models): https://www.sbert.net
- FAISS: https://github.com/facebookresearch/faiss
- sqlglot (SQL parser used by the Chapter 36 SQL guard when installed): https://github.com/tobymao/sqlglot

**Protocols, frameworks, and serving engines discussed as options**

- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- LangChain: https://python.langchain.com
- LangGraph: https://langchain-ai.github.io/langgraph/
- LlamaIndex: https://docs.llamaindex.ai
- DSPy: https://dspy.ai
- vLLM: https://docs.vllm.ai
- SGLang: https://github.com/sgl-project/sglang
- llama.cpp and the GGUF format: https://github.com/ggml-org/llama.cpp

Framework and engine APIs change between minor versions. The book teaches the underlying primitives
first so that these projects remain interchangeable implementation choices.

---

## Source acknowledgement

This book was built from the *AI Engineering Complete Study Book v2 (2026)*, which follows the topic map
of Amit Shekhar's public AI Engineering Course (https://github.com/amitshekhariitbhu/ai-engineering-course).
The chapters, code, projects, and exercises here were written anew on that foundation; the topic map and
much of the source's case material and failure catalogue shaped what the book covers.
