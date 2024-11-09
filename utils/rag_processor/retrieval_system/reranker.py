from typing import List, Tuple
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader

class Reranker:
    def __init__(self):
        self.reranker = None

    def rerank(self, query: str, corpus: List[str]) -> List[Tuple[str, float]]:
        pass


    @staticmethod
    def pretty_print_docs(docs):
        print(
            f"\n{'-' * 100}\n".join(
                [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
            )
        )
if __name__ == "__main__":
    documents = PDFPlumberLoader("CompetitionDataset/reference/finance/4.pdf").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddingsModel = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )
    retriever = FAISS.from_documents(texts, embeddingsModel).as_retriever(
        search_kwargs={"k": 20}
    )

    query = "風險要怎麼管理?"
    docs = retriever.invoke(query)
    Reranker.pretty_print_docs(docs)
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=model, top_n=1)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    compressed_docs = compression_retriever.invoke(query)
    print('='*100)
    Reranker.pretty_print_docs(compressed_docs)
