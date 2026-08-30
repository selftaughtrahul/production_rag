from .state import RAGState


class RAGNodes:

    def __init__(
        self,
        retriever,
        reranker,
        context_builder,
        llm,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.llm = llm

    def retrieve(self, state: RAGState):
        observer = state.get("observer")
        start_time = observer.on_retrieval_start() if observer else 0.0

        question = state["question"]

        query = state.get(
            "rewritten_question",
            question,
        )

        retry_count = state.get(
            "retry_count",
            0,
        )

        # Retrieve a larger candidate set
        documents = self.retriever.retrieve(
            query=query,
            top_k=20,
        )

        print(f"Retrieved {len(documents)} documents")

        # Rerank candidates
        documents = self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=5,
        )

        print(f"Reranked to {len(documents)} documents")

        if observer:
            observer.on_retrieval_end(
                start_time=start_time,
                document_count=len(documents),
            )

        return {
            "documents": documents,
            "retry_count": retry_count + 1,
        }

    def build_context(self, state: RAGState):

        documents = state.get(
            "documents",
            [],
        )

        if not documents:
            return {"context": ""}

        context = self.context_builder.build(documents)

        return {"context": context}

    def generate(self, state: RAGState):
        observer = state.get("observer")
        start_time = observer.on_generation_start() if observer else 0.0

        question = state["question"]
        context = state.get("context", "")

        if not context:
            if observer:
                observer.on_generation_end(start_time)

            return {
                "answer": ("I could not find relevant information in the documents.")
            }

        answer = self.llm.generate(
            question=question,
            context=context,
        )

        if observer:
            observer.on_generation_end(start_time)

        return {"answer": answer}

    def grade_documents(
        self,
        state: RAGState,
    ):
        observer = state.get("observer")
        documents = state.get("documents", [])

        if not documents:
            if observer:
                observer.on_documents_graded(0)

            return {"documents_relevant": False}

        relevant_documents = [
            doc
            for doc in documents
            if doc.metadata.get(
                "rerank_score",
                0.0,
            )
            >= 0.5
        ]

        if observer:
            observer.on_documents_graded(len(relevant_documents))

        return {
            "documents": relevant_documents,
            "documents_relevant": bool(relevant_documents),
        }

    def rewrite_query(
        self,
        state: RAGState,
    ):
        observer = state.get("observer")
        question = state["question"]

        if observer:
            observer.on_query_rewritten()
            observer.on_retry()

        rewritten_question = self.llm.rewrite_query(question)

        return {
            "rewritten_question": rewritten_question,
        }
