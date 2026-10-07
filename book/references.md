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
DOI or the publisher's page. Links were last checked in October 2026.

---

## Part I: AI Engineering Foundations (Chapters 1 to 3)

**AI engineering as a discipline**

- Huyen, C. (2025). *AI Engineering: Building Applications with Foundation Models.* O'Reilly. A
  book-length companion on building applications on top of foundation models, with a strong chapter on
  evaluation. Supports Chapter 1. https://www.oreilly.com/library/view/ai-engineering/9781098166298/

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
- Fedus, W., Zoph, B., and Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter
  Models with Simple and Efficient Sparsity.* JMLR 23. A clear introduction to mixture-of-experts
  routing, where only a few experts run per token. Supports Chapters 2 and 34.
  https://arxiv.org/abs/2101.03961

**Model behavior: reasoning, sycophancy, and nondeterminism**

- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement
  Learning.* Also published in Nature 645. Shows how reinforcement learning produces long reasoning
  traces, the behavior behind today's reasoning models. Supports Chapters 2 and 3.
  https://arxiv.org/abs/2501.12948
- Sharma, M., Tong, M., Korbak, T., et al. (2024). *Towards Understanding Sycophancy in Language
  Models.* ICLR. Evidence that preference training rewards agreeing with the user over being correct.
  Supports Chapter 2. https://arxiv.org/abs/2310.13548
- He, H., and Thinking Machines Lab (2025). *Defeating Nondeterminism in LLM Inference.* Connectionism
  blog. Explains why temperature 0 is still not reproducible on shared servers: batch size changes the
  order of floating-point reductions. Supports Chapter 2.
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- OpenAI. *Reasoning models.* API guide. Vendor documentation of reasoning effort and reasoning tokens,
  the controls Chapter 3 discusses. https://platform.openai.com/docs/guides/reasoning
- Anthropic. *Building with extended thinking.* API documentation. Vendor documentation of thinking
  budgets and their interaction with tool use; newer models replace fixed budgets with adaptive
  thinking, so read the current page. Supports Chapter 3.
  https://platform.claude.com/docs/en/build-with-claude/extended-thinking

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
- Agrawal, L. A., et al. (2025). *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement
  Learning.* A prompt optimizer that learns from natural-language feedback on its own rollouts.
  Supports Chapter 4. https://arxiv.org/abs/2507.19457

**Context engineering and long inputs (Chapter 5)**

- Anthropic (2025). *Effective context engineering for AI agents.* Engineering blog. A practitioner
  account of treating context as a scarce budget across system prompt, tools, and history. Supports
  Chapters 5 and 19. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Hong, K., Troynikov, A., and Huber, J. (2025). *Context Rot: How Increasing Input Tokens Impacts
  LLM Performance.* Chroma technical report. Shows quality falling with input length on deliberately
  simple tasks across 18 models. https://research.trychroma.com/context-rot
- Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., et al. (2024). *RULER: What's the Real Context Size
  of Your Long-Context Language Models?* A synthetic benchmark showing that effective context is often
  much shorter than the advertised window. https://arxiv.org/abs/2404.06654
- Modarressi, A., et al. (2025). *NoLiMa: Long-Context Evaluation Beyond Literal Matching.* A
  needle-in-a-haystack test without lexical overlap, where most models degrade sharply by 32K tokens.
  https://arxiv.org/abs/2502.05167
- Pan, Z., et al. (2024). *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic
  Prompt Compression.* Findings of ACL. Prompt compression as token classification with a small
  encoder. https://arxiv.org/abs/2403.12968

**Structured output and confidence (Chapter 6)**

- Dong, Y., Ruan, C. F., Cai, Y., Lai, R., Xu, Z., Zhao, Y., and Chen, T. (2025). *XGrammar: Flexible
  and Efficient Structured Generation Engine for Large Language Models.* MLSys. How grammar-constrained
  decoding is made fast enough for production serving. https://arxiv.org/abs/2411.15100
- Geng, S., et al. (2025). *JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language
  Models.* 10K real-world JSON schemas used to compare constrained-decoding frameworks on coverage,
  efficiency, and quality. https://arxiv.org/abs/2501.10868
- Tam, Z. R., Wu, C.-K., Tsai, Y.-L., Lin, C.-Y., Lee, H., and Chen, Y.-N. (2024). *Let Me Speak
  Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models.*
  EMNLP Industry Track. Evidence that strict formats can hurt reasoning, and why to let the model reason
  before it fills the schema. https://arxiv.org/abs/2408.02442
- Kadavath, S., et al. (2022). *Language Models (Mostly) Know What They Know.* Early evidence that
  models can estimate whether their own answers are correct. https://arxiv.org/abs/2207.05221
- Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., and Hooi, B. (2024). *Can LLMs Express Their
  Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs.* ICLR. Black-box confidence
  methods and how overconfident verbalized confidence tends to be. https://arxiv.org/abs/2306.13063

**Routing and cascades (Chapter 7)**

- Chen, L., Zaharia, M., and Zou, J. (2023). *FrugalGPT: How to Use Large Language Models While
  Reducing Cost and Improving Performance.* The paper that framed the LLM cascade.
  https://arxiv.org/abs/2305.05176
- Ong, I., et al. (2025). *RouteLLM: Learning to Route LLMs with Preference Data.* ICLR. Routers
  trained on preference data to choose between a strong and a weak model.
  https://arxiv.org/abs/2406.18665
- Hu, Q. J., et al. (2024). *RouterBench: A Benchmark for Multi-LLM Routing System.* Precomputed
  outputs from many models, so routing policies can be compared without new inference.
  https://arxiv.org/abs/2403.12031

## Part III: Embeddings and Retrieval (Chapters 8 and 9)

**Embedding models**

- Reimers, N., and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks.* EMNLP. https://arxiv.org/abs/1908.10084
- Kusupati, A., et al. (2022). *Matryoshka Representation Learning.* NeurIPS.
  https://arxiv.org/abs/2205.13147
- Muennighoff, N., Tazi, N., Magne, L., and Reimers, N. (2023). *MTEB: Massive Text Embedding
  Benchmark.* EACL. Useful for shortlisting models, never a substitute for evaluation on your data.
  https://arxiv.org/abs/2210.07316
- Enevoldsen, K., et al. (2025). *MMTEB: Massive Multilingual Text Embedding Benchmark.* ICLR. The
  multilingual successor to MTEB, with 500+ tasks across 250+ languages. Supports Chapter 8.
  https://arxiv.org/abs/2502.13595
- Weller, O., Boratko, M., Naim, I., and Lee, J. (2026). *On the Theoretical Limitations of
  Embedding-Based Retrieval.* ICLR. Shows that embedding dimension bounds which top-k result sets a
  single-vector retriever can return, with a simple dataset where strong models fail. Supports
  Chapter 8. https://arxiv.org/abs/2508.21038
- Morris, J. X., Kuleshov, V., Shmatikov, V., and Rush, A. M. (2023). *Text Embeddings Reveal (Almost)
  As Much As Text.* EMNLP. Embedding inversion recovers most short inputs exactly, so treat stored
  vectors as sensitive data. Supports Chapters 8 and 15. https://arxiv.org/abs/2310.06816

**Approximate nearest neighbor search**

- Malkov, Y. A., and Yashunin, D. A. (2018). *Efficient and Robust Approximate Nearest Neighbor Search
  Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI (published 2020).
  https://arxiv.org/abs/1603.09320
- Jégou, H., Douze, M., and Schmid, C. (2011). *Product Quantization for Nearest Neighbor Search.*
  IEEE TPAMI 33(1). https://doi.org/10.1109/TPAMI.2010.57
- Johnson, J., Douze, M., and Jégou, H. (2017). *Billion-scale Similarity Search with GPUs.* The FAISS
  paper, covering IVF and PQ at scale. https://arxiv.org/abs/1702.08734
- Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.-E., Lomeli, M., Hosseini, L.,
  and Jégou, H. (2024). *The Faiss Library.* The design and trade-offs of the library as it stands
  today, a useful map of the index families. Supports Chapter 9. https://arxiv.org/abs/2401.08281
- Jayaram Subramanya, S., Devvrit, F., Simhadri, H. V., Krishnaswamy, R., and Kadekodi, R. (2019).
  *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS. Graph search
  served from SSD when the index does not fit in memory. Supports Chapter 9.
  https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html
- Aumüller, M., Bernhardsson, E., and Faithfull, A. (2019). *ANN-Benchmarks: A Benchmarking Tool for
  Approximate Nearest Neighbor Algorithms.* Information Systems. The standard recall-versus-throughput
  comparison of ANN libraries. Supports Chapter 9. https://arxiv.org/abs/1807.05614

## Part IV: Production RAG (Chapters 10 to 15)

**Retrieval-augmented generation and dense retrieval**

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
  NeurIPS. https://arxiv.org/abs/2005.11401
- Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., and Yih, W. (2020). *Dense
  Passage Retrieval for Open-Domain Question Answering.* EMNLP. https://arxiv.org/abs/2004.04906
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and Gurevych, I. (2021). *BEIR: A Heterogeneous
  Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets and Benchmarks.
  Shows why BM25 remains a strong baseline. https://arxiv.org/abs/2104.08663
- Joren, H., Zhang, J., Ferng, C.-S., Juan, D.-C., Taly, A., and Rashtchian, C. (2025). *Sufficient
  Context: A New Lens on Retrieval Augmented Generation Systems.* ICLR. Separates "the retriever missed"
  from "the model misused good context", and uses the distinction for abstention. Supports Chapter 10.
  https://arxiv.org/abs/2411.06037
- Jin, B., Yoon, J., Han, J., and Arik, S. O. (2024). *Long-Context LLMs Meet RAG: Overcoming
  Challenges for Long Inputs in RAG.* Quality first rises, then falls, as more passages are added; hard
  negatives are the cause and reordering helps. Supports Chapter 10. https://arxiv.org/abs/2410.05983
- Li, Z., Li, C., Zhang, M., Mei, Q., and Bendersky, M. (2024). *Retrieval Augmented Generation or
  Long-Context LLMs? A Comprehensive Study and Hybrid Approach.* EMNLP Industry Track. Compares the two
  approaches on cost and quality and routes between them. Supports Chapter 10.
  https://arxiv.org/abs/2407.16833

**Document parsing and chunking (Chapter 11)**

- Auer, C., et al. (2024). *Docling Technical Report.* IBM Research. An open-source PDF conversion
  pipeline with layout analysis and table structure recognition. https://arxiv.org/abs/2408.09869
- Günther, M., Mohr, I., Williams, D. J., Wang, B., and Xiao, H. (2024). *Late Chunking: Contextual
  Chunk Embeddings Using Long-Context Embedding Models.* Embed the whole document first, then pool per
  chunk, so each chunk embedding keeps its surrounding context. https://arxiv.org/abs/2409.04701

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
- Formal, T., Lassance, C., Piwowarski, B., and Clinchant, S. (2021). *SPLADE v2: Sparse Lexical and
  Expansion Model for Information Retrieval.* Learned sparse retrieval that still runs on an inverted
  index. Supports Chapter 12. https://arxiv.org/abs/2109.10086
- Carbonell, J., and Goldstein, J. (1998). *The Use of MMR, Diversity-Based Reranking for Reordering
  Documents and Producing Summaries.* SIGIR. The original maximal marginal relevance criterion for
  trading relevance against redundancy. Supports Chapter 12. https://doi.org/10.1145/290941.291025
- Sun, W., Yan, L., Ma, X., Ren, P., Yin, D., and Ren, Z. (2023). *Is ChatGPT Good at Search?
  Investigating Large Language Models as Re-Ranking Agents.* EMNLP. Listwise LLM reranking (RankGPT) and
  its distillation into a small reranker. Supports Chapter 12. https://arxiv.org/abs/2304.09542

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
- Gao, T., Yen, H., Yu, J., and Chen, D. (2023). *Enabling Large Language Models to Generate Text with
  Citations.* EMNLP. The ALCE benchmark and its citation recall and precision metrics. Supports
  Chapter 13. https://arxiv.org/abs/2305.14627
- Saad-Falcon, J., Khattab, O., Potts, C., and Zaharia, M. (2024). *ARES: An Automated Evaluation
  Framework for Retrieval-Augmented Generation Systems.* NAACL. Small fine-tuned judges corrected with a
  few hundred human labels through prediction-powered inference. Supports Chapter 14.
  https://arxiv.org/abs/2311.09476
- Thomas, P., Spielman, S., Craswell, N., and Mitra, B. (2024). *Large Language Models Can Accurately
  Predict Searcher Preferences.* SIGIR. How Bing calibrated an LLM relevance labeler against real
  searcher feedback. Supports Chapter 14. https://arxiv.org/abs/2309.10621
- Upadhyay, S., et al. (2024). *UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing
  RELevance Assessor.* An open-source reproduction of the approach above, tested on TREC Deep Learning
  tracks. Supports Chapter 14. https://arxiv.org/abs/2406.06519

**Securing the retrieval store (Chapter 15)**

- OWASP GenAI Security Project (2025). *LLM08:2025 Vector and Embedding Weaknesses.* Access control,
  cross-tenant leakage, embedding inversion, and poisoning risks in RAG stores.
  https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/

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

**Evaluating tool use (Chapter 16)**

- Patil, S. G., Mao, H., Yan, F., Ji, C. C.-J., Suresh, V., Stoica, I., and Gonzalez, J. E. (2025). *The
  Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language
  Models.* ICML. AST-based checking of function calls, plus multi-turn and abstention tests.
  https://proceedings.mlr.press/v267/patil25a.html
- Yao, S., Shinn, N., Razavi, P., and Narasimhan, K. (2024). *τ-bench: A Benchmark for Tool-Agent-User
  Interaction in Real-World Domains.* Simulated users, domain policies, and the pass^k reliability
  metric. Supports Chapters 16 and 19. https://arxiv.org/abs/2406.12045

**Durable workflows (Chapter 17)**

- Horthy, D., and HumanLayer (2025). *12-Factor Agents.* A design guide for production agents: own
  your prompts, context, and control flow, and make pause and resume explicit.
  https://github.com/humanlayer/12-factor-agents
- OpenAI (2025). *A practical guide to building agents.* Vendor guide covering when to use an agent,
  orchestration patterns, and guardrails. Supports Chapters 17 and 19.
  https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf

## Part VI: Agents (Chapters 19 to 23)

**Agent loops and architectures**

- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. (2023). *ReAct:
  Synergizing Reasoning and Acting in Language Models.* ICLR. https://arxiv.org/abs/2210.03629
- Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., and Yao, S. (2023). *Reflexion:
  Language Agents with Verbal Reinforcement Learning.* NeurIPS. https://arxiv.org/abs/2303.11366
- Madaan, A., et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback.* NeurIPS. A
  generate, critique, revise loop; read alongside Chapter 20's caution about self-critique without new
  evidence. https://arxiv.org/abs/2303.17651
- Erdogan, L. E., Lee, N., Kim, S., Moon, S., Furuta, H., Anumanchipalli, G., Keutzer, K., and
  Gholami, A. (2025). *Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks.* ICML. A
  separate planner and executor, with synthetic plan data to train the planner. Supports Chapter 20.
  https://arxiv.org/abs/2503.09572
- Anthropic (2024). *Building effective agents.* Engineering blog. Workflows versus agents, and the
  case for simple, composable patterns before frameworks. Supports Chapters 1, 19, and 23.
  https://www.anthropic.com/engineering/building-effective-agents

**Memory**

- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., and Bernstein, M. S. (2023).
  *Generative Agents: Interactive Simulacra of Human Behavior.* UIST. Memory retrieval by recency,
  importance, and relevance, plus reflection. https://arxiv.org/abs/2304.03442
- Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., and Gonzalez, J. E. (2023).
  *MemGPT: Towards LLMs as Operating Systems.* Tiered memory with explicit paging.
  https://arxiv.org/abs/2310.08560
- Sumers, T. R., Yao, S., Narasimhan, K., and Griffiths, T. L. (2024). *Cognitive Architectures for
  Language Agents.* TMLR. The CoALA framework: working, episodic, semantic, and procedural memory as
  parts of one agent design. Supports Chapter 21. https://arxiv.org/abs/2309.02427
- Chhikara, P., Khant, D., Aryan, S., Singh, T., and Yadav, D. (2025). *Mem0: Building
  Production-Ready AI Agents with Scalable Long-Term Memory.* Extract, consolidate, and retrieve memories
  instead of replaying whole histories; reports latency and token savings. Supports Chapter 21.
  https://arxiv.org/abs/2504.19413
- Xu, W., Mei, K., Gao, H., Tan, J., Liang, Z., and Zhang, Y. (2025). *A-MEM: Agentic Memory for LLM
  Agents.* Memories stored as linked notes that the agent itself organizes. Supports Chapter 21.
  https://arxiv.org/abs/2502.12110
- AGENTS.md. *A simple, open format for guiding coding agents.* A repository-level instruction file,
  now stewarded by the Agentic AI Foundation, and an example of procedural memory kept in version
  control. Supports Chapter 21. https://agents.md

**Long-term memory benchmarks**

- Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., and Yu, D. (2025). *LongMemEval: Benchmarking Chat
  Assistants on Long-Term Interactive Memory.* ICLR. Tests extraction, multi-session and temporal
  reasoning, knowledge updates, and abstention over long chat histories. Supports Chapter 21.
  https://arxiv.org/abs/2410.10813

**Multi-agent systems**

- Wu, Q., et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.*
  https://arxiv.org/abs/2308.08155
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. (2023). *Improving Factuality and
  Reasoning in Language Models through Multiagent Debate.* https://arxiv.org/abs/2305.14325
- Cemri, M., et al. (2025). *Why Do Multi-Agent LLM Systems Fail?* The MAST taxonomy of 14 failure
  modes, most of them design and coordination problems rather than model limits. Supports Chapter 22.
  https://arxiv.org/abs/2503.13657
- Anthropic (2025). *How we built our multi-agent research system.* Engineering blog. An
  orchestrator with parallel subagents, and the token cost and evaluation lessons that came with it.
  Supports Chapter 22. https://www.anthropic.com/engineering/multi-agent-research-system
- Yan, W., and Cognition (2025). *Don't Build Multi-Agents.* Blog post. The opposing view: split
  context causes conflicting decisions, so prefer a single-threaded agent. Read it next to the previous
  entry. Supports Chapter 22. https://cognition.ai/blog/dont-build-multi-agents

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

**Judges, error bars, and online tests**

- Shankar, S., Zamfirescu-Pereira, J. D., Hartmann, B., Parameswaran, A. G., and Arawjo, I. (2024).
  *Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human
  Preferences.* Introduces "criteria drift": grading outputs is how people discover their criteria.
  Supports Chapters 1 and 24. https://arxiv.org/abs/2404.12272
- Miller, E. (2024). *Adding Error Bars to Evals: A Statistical Approach to Language Model
  Evaluations.* Standard errors, clustered questions, and paired comparisons for eval scores. Supports
  Chapter 24. https://arxiv.org/abs/2411.00640
- Ye, J., et al. (2025). *Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge.* ICLR. A catalog
  of 12 judge biases and a framework for measuring each. Supports Chapter 24.
  https://arxiv.org/abs/2410.02736
- Thakur, A. S., Choudhary, K., Ramayapally, V. S., Vaidyanathan, S., and Hupkes, D. (2025). *Judging
  the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges.* GEM Workshop. Even the best
  judges diverge from humans and lean toward leniency. Supports Chapter 24.
  https://arxiv.org/abs/2406.12624
- Gu, J., et al. (2024). *A Survey on LLM-as-a-Judge.* A broad map of judge designs, bias mitigations,
  and ways to test judge reliability. Supports Chapter 24. https://arxiv.org/abs/2411.15594
- Chen, M., et al. (2021). *Evaluating Large Language Models Trained on Code.* The Codex paper; defines
  HumanEval and the unbiased pass@k estimator. Supports Chapter 25. https://arxiv.org/abs/2107.03374
- Johari, R., Koomen, P., Pekelis, L., and Walsh, D. (2022). *Always Valid Inference: Continuous
  Monitoring of A/B Tests.* Operations Research 70(3). How to peek at a running experiment without
  inflating false positives. Supports Chapter 25. https://arxiv.org/abs/1512.04922

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

**Injection-resistant agent design**

- Willison, S. (2023). *The Dual LLM pattern for building AI assistants that can resist prompt
  injection.* Blog post. A privileged model that never sees untrusted text, and a quarantined model
  that has no tools. Supports Chapter 26. https://simonwillison.net/2023/Apr/25/dual-llm-pattern/
- Debenedetti, E., et al. (2025). *Defeating Prompt Injections by Design.* CaMeL: the control flow is
  derived from the trusted query, and capabilities track where untrusted data may flow. Supports
  Chapter 26. https://arxiv.org/abs/2503.18813
- Beurer-Kellner, L., et al. (2025). *Design Patterns for Securing LLM Agents against Prompt
  Injections.* Six patterns, such as plan-then-execute and action selectors, that give up some
  generality in exchange for injection resistance. Supports Chapters 4 and 26.
  https://arxiv.org/abs/2506.08837

**Guardrail models and toolkits (Chapter 27)**

- Inan, H., et al. (2023). *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations.*
  An instruction-tuned classifier for prompts and responses against a customizable risk taxonomy.
  https://arxiv.org/abs/2312.06674
- Zeng, W., et al. (2024). *ShieldGemma: Generative AI Content Moderation Based on Gemma.* Open
  safety classifiers trained largely on synthetic data. https://arxiv.org/abs/2407.21772
- Rebedea, T., Dinu, R., Sreedhar, M., Parisien, C., and Cohen, J. (2023). *NeMo Guardrails: A Toolkit
  for Controllable and Safe LLM Applications with Programmable Rails.* EMNLP (demo). Programmable input,
  output, and dialogue rails around any model. https://arxiv.org/abs/2310.10501

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
- Beyer, B., Murphy, N. R., Rensin, D. K., Kawahara, K., and Thorne, S. (eds.) (2018). *The Site
  Reliability Workbook*, Chapter 5: *Alerting on SLOs.* O'Reilly. Burn-rate alerts built step by step
  from an SLO. Supports Chapter 29. https://sre.google/workbook/alerting-on-slos/
- Yanacek, D. *Using load shedding to avoid overload.* Amazon Builders' Library. Rejecting excess work
  early so accepted requests keep their latency. Supports Chapter 29.
  https://aws.amazon.com/builders-library/using-load-shedding-to-avoid-overload/
- Fowler, M. *CircuitBreaker.* https://martinfowler.com/bliki/CircuitBreaker.html
- Kleppmann, M. (2017). *Designing Data-Intensive Applications.* O'Reilly. Queues, idempotence,
  delivery guarantees, and event logs. A second edition by Kleppmann, M., and Riccomini, C. was
  published in 2026. https://www.oreilly.com/library/view/designing-data-intensive-applications/9781098119058/

**Engineering practices**

- Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS. Still the
  best short account of why the model is the small part of the system.
- Evans, E. (2003). *Domain-Driven Design.* Addison-Wesley. The anti-corruption layer pattern used for
  provider abstraction.
- Nygard, M. (2011). *Documenting Architecture Decisions.* The original proposal for ADRs.
  https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
- Zinkevich, M. *Rules of Machine Learning: Best Practices for ML Engineering.* Google for Developers.
  Forty-three rules, most of them about pipelines, metrics, and launching simple first. Supports
  Chapter 32. https://developers.google.com/machine-learning/guides/rules-of-ml/

**Caching (Chapter 30)**

- Gim, I., Chen, G., Lee, S., Sarda, N., Khandelwal, A., and Zhong, L. (2024). *Prompt Cache: Modular
  Attention Reuse for Low-Latency Inference.* MLSys. Reusing precomputed attention states for shared
  prompt segments, the idea behind provider prompt caching. https://arxiv.org/abs/2311.04934
- Bang, F. (2023). *GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers
  and Cost Savings.* NLP-OSS Workshop. A reference design for embedding-keyed response caching.
  https://aclanthology.org/2023.nlposs-1.24/

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
- Biderman, D., et al. (2024). *LoRA Learns Less and Forgets Less.* TMLR. LoRA trails full fine-tuning
  on new domains but preserves more of the base model's other skills. https://arxiv.org/abs/2405.09673
- Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
  Models.* Introduces GRPO, the group-relative policy optimization used in later reasoning models.
  https://arxiv.org/abs/2402.03300
- OpenAI. *Reinforcement fine-tuning.* API guide. Vendor documentation of grader-driven fine-tuning,
  an example of reinforcement fine-tuning offered as a managed service.
  https://platform.openai.com/docs/guides/reinforcement-fine-tuning

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

**Serving: mixture of experts and multi-adapter serving (Chapter 34)**

- Jiang, A. Q., et al. (2024). *Mixtral of Experts.* An open sparse mixture-of-experts model: 47B
  parameters, of which about 13B are active per token. https://arxiv.org/abs/2401.04088
- DeepSeek-AI (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language
  Model.* Introduces multi-head latent attention, which compresses the KV cache.
  https://arxiv.org/abs/2405.04434
- Sheng, Y., et al. (2024). *S-LoRA: Serving Thousands of Concurrent LoRA Adapters.* MLSys. Unified
  paging of adapters and KV cache to serve many fine-tunes on one base model.
  https://arxiv.org/abs/2311.03285
- Chen, L., et al. (2024). *Punica: Multi-Tenant LoRA Serving.* MLSys. A batched kernel that runs
  requests for different adapters in one batch. https://arxiv.org/abs/2310.18547
- Qin, R., et al. (2024). *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.*
  The production design behind Kimi: disaggregated prefill and decode with a KV cache pool across CPU,
  DRAM, and SSD. A revised version appeared at FAST 2025. https://arxiv.org/abs/2407.00079

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
- Aminian, A., and Sheng, H. (2024). *Generative AI System Design Interview.* ByteByteGo. Worked
  system design cases for generative AI products, in the interview format of Chapter 35.
- Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., and Manning, C. D. (2024). *RAPTOR:
  Recursive Abstractive Processing for Tree-Organized Retrieval.* ICLR. A tree of recursive summaries
  for questions that need a whole document. Supports Chapter 37. https://arxiv.org/abs/2401.18059
- Faysse, M., et al. (2025). *ColPali: Efficient Document Retrieval with Vision Language Models.*
  ICLR. Retrieves page images directly with late interaction, skipping text extraction. Supports
  Chapter 37. https://arxiv.org/abs/2407.01449
- Jin, B., et al. (2025). *Search-R1: Training LLMs to Reason and Leverage Search Engines with
  Reinforcement Learning.* Trains the model to decide when and what to search during reasoning.
  Supports Chapter 37. https://arxiv.org/abs/2503.09516
- Anthropic (2025). *Equipping agents for the real world with Agent Skills.* Engineering blog. Folders
  of instructions and scripts that an agent loads on demand, a packaging format for procedural
  knowledge. Supports Chapter 38.
  https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills

---

## Specifications and standards

Check the current version of each before relying on details; several are revised more than once a
year.

- **Model Context Protocol (MCP).** Specification, concepts, and SDKs. The current revision as of this
  edition is 2026-07-28, which makes the protocol core stateless and retires the initialize handshake;
  the previous revision was 2025-11-25. Read the changelog of whichever revision your SDK targets.
  Supports Chapter 18. https://modelcontextprotocol.io/specification/2026-07-28/changelog
- **MCP Registry.** The official catalog and API for publicly available MCP servers.
  https://registry.modelcontextprotocol.io
- **Agent2Agent (A2A) Protocol.** An open protocol, now under the Linux Foundation, for agents to
  discover each other through Agent Cards and delegate tasks over HTTP, JSON-RPC, gRPC, or SSE.
  Supports Chapters 18 and 22. https://a2a-protocol.org/latest/specification/
- **JSON-RPC 2.0 Specification.** The message format MCP builds on. https://www.jsonrpc.org/specification
- **RFC 9728, OAuth 2.0 Protected Resource Metadata (2025).** How a client discovers which
  authorization server protects a resource; MCP authorization relies on it.
  https://www.rfc-editor.org/rfc/rfc9728
- **RFC 8707, Resource Indicators for OAuth 2.0 (2020).** Lets a client name the resource a token is
  for, so the token can be restricted to that one server. https://www.rfc-editor.org/rfc/rfc8707
- **OpenTelemetry Semantic Conventions for Generative AI.** Standard attribute names for model calls,
  token usage, agent operations, and MCP. In June 2026 the conventions moved to their own repository,
  where they are still marked Development; the old pages on opentelemetry.io now redirect.
  https://github.com/open-telemetry/semantic-conventions-genai
- **W3C Trace Context.** How trace identifiers propagate across services.
  https://www.w3.org/TR/trace-context/
- **OWASP Top 10 for LLM Applications 2025.** The current edition, published in November 2024 by the
  OWASP GenAI Security Project; it adds System Prompt Leakage and Vector and Embedding Weaknesses and
  widens Denial of Service into Unbounded Consumption.
  https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/
- **OWASP Agentic AI: Threats and Mitigations.** The first guide from the OWASP Agentic Security
  Initiative, a threat-model reference for tool use, memory, and multi-agent delegation. Supports
  Chapter 26. https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/
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
- LiteLLM (one API across many model providers; Chapter 3): https://docs.litellm.ai
- OpenAI Agents SDK (Chapter 23): https://openai.github.io/openai-agents-python/
- Google Agent Development Kit (ADK) (Chapter 23): https://google.github.io/adk-docs/
- Pydantic AI (Chapter 23): https://ai.pydantic.dev
- Claude Agent SDK (Chapter 23): https://platform.claude.com/docs/en/agent-sdk/overview

**Durable execution engines (Chapters 17 and 38)**

- Temporal: https://docs.temporal.io
- AWS Step Functions: https://docs.aws.amazon.com/step-functions/latest/dg/welcome.html
- DBOS (durable workflows on Postgres): https://docs.dbos.dev
- Restate: https://docs.restate.dev

Framework and engine APIs change between minor versions. The book teaches the underlying primitives
first so that these projects remain interchangeable implementation choices.

---

## Source acknowledgement

This book was built from the *AI Engineering Complete Study Book v2 (2026)*, which follows the topic map
of Amit Shekhar's public AI Engineering Course (https://github.com/amitshekhariitbhu/ai-engineering-course).
The chapters, code, projects, and exercises here were written anew on that foundation; the topic map and
much of the source's case material and failure catalogue shaped what the book covers.
