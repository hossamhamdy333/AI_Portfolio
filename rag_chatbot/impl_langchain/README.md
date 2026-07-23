# RAG Chatbot — LangChain Implementation

Same corpus (XL-Sum Arabic), same 400-question synthetic eval set as `impl_vanilla/`.
The point of this version is a controlled comparison, not a re-skin: everything is
identical to vanilla except the retrieval/generation is built with LangChain.

## What's different from `impl_vanilla/`

- Retrieval built with LangChain's `ParentDocumentRetriever` (small chunks for
  search, larger parent chunks passed to the LLM) instead of a single flat
  fixed-size chunk — something vanilla doesn't do, so this isn't just the same
  pipeline copy-pasted into LCEL syntax.
- Generation wired as an LCEL chain (`retriever | prompt | llm | citation_parser`).
- Same citation-verification step as vanilla — re-parse the model's `[N]` tags
  and map them back to source article IDs, rather than trusting LangChain's
  output at face value.

## Structure

```
impl_langchain/
├── src/
│   ├── retriever.py     # ParentDocumentRetriever setup over XLSum Arabic
│   ├── chain.py         # LCEL chain: retrieve → prompt → generate → cite
│   └── generation.py    # citation parsing/verification (same approach as vanilla)
├── notebooks/
│   └── 04_rag_pipeline.ipynb
└── README.md
```

## Structure (actual)

```
impl_langchain/
├── src/
│   ├── retriever.py       # ParentDocumentRetriever over xlsum_arabic_clean.parquet
│   ├── chain.py            # LCEL: retrieve -> rerank (same cross-encoder as vanilla) -> generate
│   ├── generation.py       # [N] citation parsing/verification, same correctness definition as vanilla
│   └── eval_langchain.py   # eval loop -- loads vanilla's eval set via shared/, scores via shared/metrics.py
├── notebooks/               # empty -- port the wire-up sketch in eval_langchain.py into a notebook
└── README.md
```

## Status

Code written, **not yet run**. Next steps:
1. `pip install langchain langchain-chroma langchain-huggingface langchain-google-genai`
2. Point `retriever.py` at the actual `xlsum_arabic_clean.parquet` (copy/symlink from `impl_vanilla/data/processed/`)
3. Run `eval_langchain.py`'s wire-up against the real 100-question sample, fill in `COMPARISON.md`
