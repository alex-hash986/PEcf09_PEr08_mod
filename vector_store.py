"""
Модуль работы с векторным хранилищем ChromaDB для OpenAI Embeddings.
Использует семантическое разбиение текста на чанки по абзацам и предложениям.
"""

import os
import re
from typing import Any, Dict, List, Optional
import chromadb
from openai import OpenAI


class VectorStore:
    """Векторное хранилище на основе ChromaDB с OpenAI embeddings."""

    def __init__(
        self,
        collection_name: str = "api_rag_collection",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY не установлен")

        self.openai_client = OpenAI(api_key=resolved_key, base_url=resolved_url)
        self.client = chromadb.PersistentClient(path=persist_directory)

        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Коллекция '{collection_name}' загружена. Чанков: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"Создана новая коллекция '{collection_name}'")

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Умное разбиение текста на чанки по абзацам и предложениям."""
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
            elif current_chunk:
                chunks.append(current_chunk)
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = f"{overlap_text}\n\n{paragraph}" if overlap_text else paragraph
            else:
                if len(paragraph) > chunk_size:
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)

        return [chunk for chunk in chunks if len(chunk) >= 50]

    @staticmethod
    def _get_overlap_text(text: str, overlap_size: int) -> str:
        """Извлечение перекрытия по границам предложений."""
        if len(text) <= overlap_size:
            return text

        overlap_candidate = text[-overlap_size:]
        sentence_starts = [". ", "! ", "? ", "\n"]
        best_start = 0

        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)

        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        return overlap_candidate.strip()

    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """Разбиение длинного абзаца по предложениям."""
        sentences = re.split(r"([.!?]+\s+)", paragraph)
        full_sentences: List[str] = []

        for i in range(0, len(sentences) - 1, 2):
            full_sentences.append(sentences[i] + sentences[i + 1])
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])

        chunks: List[str] = []
        current_chunk = ""

        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = f"{overlap_text} {sentence}" if overlap_text else sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Создание эмбеддингов через OpenAI API."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def load_documents(self, file_path: str) -> None:
        """Загрузка документов из файла в ChromaDB."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден")

        if self.collection.count() > 0:
            print("Документы уже загружены в коллекцию")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = self._chunk_text(text)
        print(f"Текст разбит на {len(chunks)} чанков")

        ids = [f"doc_{i}" for i in range(len(chunks))]
        embeddings = self._create_embeddings(chunks)

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
        )
        print(f"Загружено {len(chunks)} чанков в коллекцию '{self.collection_name}'")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Поиск релевантных чанков по запросу."""
        if self.collection.count() == 0:
            return []

        query_embedding = self._create_embeddings([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
        )

        documents: List[Dict[str, Any]] = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                documents.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None,
                    }
                )
        return documents

    def get_collection_stats(self) -> Dict[str, Any]:
        """Получение статистики коллекции."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model,
        }