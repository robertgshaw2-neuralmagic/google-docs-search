import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EmbeddingModel:
    def __init__(self, model="text-embedding-ada-002"):
        api_key = os.environ["OPENAI_API_KEY"]

        self._text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=model,
        )

        self._model = OpenAIEmbeddings(
            openai_api_key=api_key,
            model=model,
        )
    
    def _split_document(self, document):
        return [doc.page_content for doc in self._text_splitter.create_documents([document])]
    
    def embed_documents(self, documents):
        document_chunks = []
        chunk_ids = []
        indexes = []
        for i, document in enumerate(documents):
            chunks = self._split_document(document)
            document_chunks += chunks
            indexes += ([i] * len(chunks))
            chunk_ids += range(len(chunks))
            
        embeddings = self._model.embed_documents(document_chunks)
        return document_chunks, embeddings, indexes, chunk_ids
    
    def embed_query(self, query):
        assert(len(query) < 5000)
        return self._model.embed_query(query)