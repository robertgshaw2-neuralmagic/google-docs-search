import langchain
from apis.embedding_model import EmbeddingModel

class QAChain:
    def __init__(self):

        self._embedding_db = langchain.vectorstores.Chroma(
            collection_name="google_docs",
            persist_directory="./.chroma",
            embedding_function=EmbeddingModel()._model,
            )
        
        self._retriever = self._embedding_db.as_retriever()
        self._retriever.search_kwargs['distance_metric'] = 'cos'
        self._retriever.search_kwargs['k'] = 4
        
        self._chain = langchain.chains.RetrievalQA.from_chain_type(
            langchain.chat_models.ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            chain_type="stuff",
            retriever=self._retriever, 
            return_source_documents=False)

    def qa(self, query):
        return (self._chain({"query":query}))
