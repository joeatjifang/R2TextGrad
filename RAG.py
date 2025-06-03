import numpy as np
import os
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sklearn.metrics.pairwise import cosine_similarity
from engine import get_engine
from engine.openai import ChatOpenAI
from engine.deepseek import ChatDeepSeek
import json

class OpenAIEmbeddingClient:
    def __init__(self, model_string: str = "text-embedding-ada-002"):
        self.engine = get_engine("gpt-4")
        self.model = model_string

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException,))
    )
    def get_embedding(self, text: str) -> List[float]:
        """Get text embeddings"""
        try:
            response = self.engine.client.embeddings.create(
                model=self.model,
                input=[text]
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Failed to get embedding: {e}")
            raise


class AnnualReportProcessor:
    """Annual Report Processor for listed companies"""

    def __init__(self, embedding_client: OpenAIEmbeddingClient, max_chunk_size: int = 1000):
        self.embedding_client = embedding_client
        self.max_chunk_size = max_chunk_size

    def split_text(self, text: str) -> List[str]:
        """Split text into appropriate sized chunks"""
        if not text:
            return []

        # Simple text splitting based on periods
        sentences = text.replace('\n', ' ').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                sub_sentences = self._split_long_sentence(sentence)
                for sub_sent in sub_sentences:
                    if len(current_chunk) + len(sub_sent) + 1 > self.max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = sub_sent + "."
                        else:
                            chunks.append(sub_sent + ".")
                            current_chunk = ""
                    else:
                        current_chunk += sub_sent + "."
                continue

            current_chunk += sentence + "."

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split long sentences"""
        if len(sentence) <= self.max_chunk_size:
            return [sentence]

        sub_sentences = []
        parts = sentence.split(',')
        current_part = ""

        for part in parts:
            if len(current_part) + len(part) + 1 > self.max_chunk_size:
                if current_part:
                    sub_sentences.append(current_part)
                current_part = part + ","
            else:
                current_part += part + ","

        if current_part:
            sub_sentences.append(current_part[:-1])

        final_subs = []
        for sub in sub_sentences:
            if len(sub) > self.max_chunk_size:
                for i in range(0, len(sub), self.max_chunk_size):
                    final_subs.append(sub[i:i + self.max_chunk_size])
            else:
                final_subs.append(sub)

        return final_subs

    def process_report(self, file_path: str, company: str, year: str) -> List[Dict[str, Any]]:
        """Process a single annual report file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            chunks = self.split_text(content)
            documents = []

            for i, chunk in enumerate(chunks):
                embedding = self.embedding_client.get_embedding(chunk)
                document = {
                    "id": f"{os.path.basename(file_path)}_{i}",
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": {
                        "source": file_path,
                        "company": company,
                        "year": year,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                }
                documents.append(document)

            return documents
        except Exception as e:
            print(f"Failed to process file: {file_path}, error: {e}")
            return []


class KnowledgeBase:
    """RAG Knowledge Base"""

    def __init__(self, storage_path: str, company: str, year: str):
        self.storage_path = storage_path
        self.company = company
        self.year = year
        self.documents = []
        self.storage_path = self._get_storage_path()

        if os.path.exists(storage_path):
            self.load()

    def _get_storage_path(self) -> str:
        """Get knowledge base storage path"""
        os.makedirs(self.storage_path, exist_ok=True)
        return os.path.join(self.storage_path, f"{self.company}_{self.year}_knowledge_base.json")

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to knowledge base"""
        self.documents.extend(documents)

    def save(self):
        """Save knowledge base to disk"""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            print(f"Knowledge base saved to {self.storage_path}")
        except Exception as e:
            print(f"Failed to save knowledge base: {e}")

    def load(self):
        """Load knowledge base from disk"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print(f"Knowledge base loaded from {self.storage_path}")
        except Exception as e:
            print(f"Failed to load knowledge base: {e}")
            self.documents = []


class RAGPromptGenerator:
    """RAG-based Prompt Generator"""

    def __init__(self, knowledge_base_dir: str, company: str, year: str):
        self.knowledge_base_dir = knowledge_base_dir
        self.company = company
        self.year = year
        self.documents = []
        self.embeddings = None
        self.embedding_client = OpenAIEmbeddingClient()
        self.load_knowledge_base()

    def _build_embeddings_index(self):
        """Build embedding vector index for fast retrieval"""
        if not self.documents:
            return
        self.embeddings = np.array([doc["embedding"] for doc in self.documents])

    def retrieve_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve documents relevant to the query"""
        if not self.documents or self.embeddings is None:
            raise ValueError("Knowledge base is empty or not loaded")

        query_embedding = self.embedding_client.get_embedding(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        relevant_docs = [self.documents[i] for i in top_indices]

        for i, doc in enumerate(relevant_docs):
            doc["similarity"] = float(similarities[top_indices[i]])

        return relevant_docs

    def generate_prompt(self, query: str, top_k: int = 3, system_prompt: Optional[str] = None) -> str:
        """Generate prompt with relevant knowledge"""
        relevant_docs = self.retrieve_relevant_documents(query, top_k)

        context = "\n\n".join([
            f"Related Content #{i + 1} (similarity: {doc['similarity']:.4f}):\n{doc['content']}"
            for i, doc in enumerate(relevant_docs)
        ])

        if system_prompt:
            full_prompt = f"{system_prompt}\n\nBased on the following information from {self.company}'s {self.year} annual report related to '{query}', please answer the question:\n{context}\n\nQuestion: {query}"
        else:
            full_prompt = f"Based on the following information from {self.company}'s {self.year} annual report related to '{query}', please answer the question:\n{context}\n\nQuestion: {query}"

        return full_prompt


class RAGRetriever:
    """RAG-based Document Retriever with support for various document types"""

    def __init__(self, documents: List[str], model_string: str = "text-embedding-ada-002"):
        self.documents = documents
        self.embedding_client = OpenAIEmbeddingClient(model_string)
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize document embeddings"""
        self.embeddings = np.array([
            self.embedding_client.get_embedding(doc) 
            for doc in self.documents
        ])

    def get_relevant_passages(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve the k most relevant passages for a query using cosine similarity
        
        Args:
            query: The search query
            k: Number of passages to retrieve (default: 3)
            
        Returns:
            List of k most relevant document passages
        """
        if not self.documents or self.embeddings is None:
            raise ValueError("Document collection is empty or embeddings not initialized")

        # Get query embedding
        query_embedding = self.embedding_client.get_embedding(query)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top k indices
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        # Return relevant documents
        return [self.documents[i] for i in top_k_indices]

    def get_similar_documents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Get similar documents with similarity scores
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of dicts containing document content and similarity scores
        """
        query_embedding = self.embedding_client.get_embedding(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'content': self.documents[idx],
                'similarity': float(similarities[idx])
            })
        
        return results


def main():
    embedding_client = OpenAIEmbeddingClient()
    processor = AnnualReportProcessor(embedding_client, 1000)
    
    year_list = ['2017', '2018', '2019', '2020', '2021']
    for year in tqdm(year_list):
        data_path = 'RawData/Management_Discussion_Analysis/MDA_ALL/External_MDA_ALL/' + year + '/text/'
        files = os.listdir(data_path)
        files_sorted = [file for file in files if '-12-31' in file]
        for file in files_sorted:
            temp_company, temp_year = file[0:6], file[7:11]
            documents = processor.process_report(data_path+file, temp_company, temp_year)
            knowledge_base = KnowledgeBase('KnowledgeBase', temp_company, temp_year)
            knowledge_base.add_documents(documents)
            knowledge_base.save()
            print(f"Knowledge base for {temp_company}_{temp_year} built, containing {len(knowledge_base.documents)} document chunks")

    # Initialize the engine
    engine = ChatDeepSeek(
        model_string="deepseek-r1",
        temperature=0.7
    )

    # Generate response
    response = engine.generate("What is quantum computing?")

    # Or use it with the optimizer
    optimizer = R2TextualGradientDescent(
        parameters=[prompt_var],
        engine=engine,
        num_trials=5,
        gradient_memory=3
    )

if __name__ == '__main__':
    main()
