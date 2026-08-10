"""LCEL chain: retrieve (parent-doc) -> rerank -> prompt -> generate.

Keeps the same cross-encoder reranker step impl_vanilla uses (retrieval
alone isn't the comparison point here -- reranking is identical across
implementations by design, so any metric delta traces back to the
retriever architecture, not a second confounding variable).
"""

from operator import itemgetter
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# Same instruction shape as impl_vanilla's Gemini prompt: numbered sources,
# forced [N] citation tags, answer must come only from provided context.
PROMPT_TEMPLATE = """أنت مساعد يجيب على الأسئلة بالاعتماد فقط على المصادر المرفقة أدناه.
استشهد بكل معلومة تستخدمها بوضع رقم المصدر بين قوسين مربعين، مثل [1].
إذا لم تكن الإجابة موجودة في المصادر، قل إنك لا تعرف.

المصادر:
{context}

السؤال: {question}

الإجابة (مع الاستشهادات [N]):"""


def format_context(reranked_docs):
    return "\n\n".join(
        f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(reranked_docs)
    )


def build_chain(retriever, reranker, llm_model_name, top_k_rerank=5, temperature=0.2):
    """reranker: same cross-encoder object impl_vanilla's retrieval.py uses
    (sentence-transformers CrossEncoder), passed in rather than reconstructed
    here so both implementations are provably using the identical model.
    """
    llm = ChatGoogleGenerativeAI(
        model=llm_model_name, temperature=temperature, google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def retrieve_and_rerank(question: str):
        candidates = retriever.invoke(question)
        pairs = [[question, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k_rerank]]

    # Stage 1: fan out -- retrieve+rerank on one branch, pass the raw
    # question through unchanged on the other.
    retrieve_stage = RunnableParallel(
        reranked_docs=RunnableLambda(retrieve_and_rerank),
        question=RunnablePassthrough(),
    )

    # Stage 2: build the answer from stage 1's output, while also carrying
    # the doc list through untouched so callers get (answer, docs) back,
    # same interface impl_vanilla's generate_answer() returns.
    generate_stage = RunnableParallel(
        answer=(
            RunnableLambda(lambda x: {"context": format_context(x["reranked_docs"]), "question": x["question"]})
            | prompt
            | llm
            | StrOutputParser()
        ),
        docs=itemgetter("reranked_docs"),
    )

    chain = retrieve_stage | generate_stage

    def run(question: str):
        result = chain.invoke(question)
        return result["answer"], result["docs"]

    return run
