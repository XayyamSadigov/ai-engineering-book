# AI Engineering: From Software Engineer to Production AI Engineer

A complete, self-contained textbook and field guide: 39 chapters, a shared Python library, six
projects, and a production-grade capstone. Read it from the first chapter to the capstone and you will
be able to design, implement, evaluate, secure, deploy, observe, and improve production applications
built on large language models.

**Read online:** <https://xayyamsadigov.github.io/ai-engineering-book/>

- [About the book and how it is organized](book/README.md)
- [Learning roadmap](book/00-learning-roadmap.md) — full, accelerated and role-based reading paths
- [Chapters](book/chapters/) · [Exercises](book/exercises/) · [Solutions](book/solutions/)
- [Projects](book/projects/) · [Capstone: Northwind Assist](book/capstone/northwind-assist/)
- [The whole book in one file](book/AI_ENGINEERING_BOOK.md)

## Running the code

All code targets Python 3.11+ and runs offline in tests (fake model and embedding clients). Real
providers are enabled through environment variables documented in each project's `.env.example`.

```bash
cd book/projects/aie_core && pip install -e .          # the shared library
cd ../p3-rag-assistant && pip install -e . && pytest   # any project
book/tools/verify_code.sh                              # compile and test everything
```

See each project's README for its own setup.

## Building the site locally

```bash
pip install mkdocs-material
mkdocs serve
```

## License

Text: [CC BY 4.0](LICENSE-CONTENT.md). Code: [MIT](LICENSE).
