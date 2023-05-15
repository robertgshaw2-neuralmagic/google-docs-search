import chromadb

class EmbeddingDB:
    def __init__(self, collection_name="google_docs", reset_database=False):
        self._client = chromadb.Client(
            chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./.chroma/"                         
        ))

        self._collection_name = collection_name
        if reset_database:
            self.delete_collection()
        self.load_collection()
    
    def load_collection(self):
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

    def delete_collection(self):
        self._client.delete_collection(name=self._collection_name)
        self._collection = None   

    def _format_metadata(self, doc_metadatas, indexes, chunk_ids):
        metadatas = []
        ids = []
        for index, chunk_id in zip(indexes, chunk_ids):
            gdoc_id = doc_metadatas[index]["gdoc_id"]
            gdoc_title = doc_metadatas[index]["gdoc_title"]
            metadatas.append({
                'gdoc_id': gdoc_id,
                'title': gdoc_title,
                'chunk_id': chunk_id
            })
            ids.append(f"{gdoc_id}__chunk_id={chunk_id}")

        return ids, metadatas
        
    def insert_documents(self, document_chunks, embeddings, doc_metadatas, indexes, chunk_ids):
        ids, metadatas = self._format_metadata(doc_metadatas, indexes, chunk_ids)

        self._collection.upsert(
            documents=document_chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
    
    def get_items_by_gdoc_id(self, gdoc_id):
        return self._collection.get(
            where={"gdoc_id": gdoc_id}
        )