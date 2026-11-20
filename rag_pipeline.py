"""
Основной RAG pipeline для OpenAI API с SQLite-кэшированием и логированием метрик.
Поток: запрос -> кэш -> vector search -> OpenAI LLM -> кэш -> логирование (logs.db).
"""

import os
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI

from vector_store import VectorStore
from cache import RAGCache
from db_logger import DatabaseLogger


class RAGPipeline:
    """Основной pipeline для RAG-системы на базе OpenAI API."""

    def __init__(
        self,
        collection_name: str = "api_rag_collection",
        cache_db_path: str = "api_rag_cache.db",
        logs_db_path: str = "logs.db",
        data_file: str = "data/docs.txt",
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0.3,
        enable_logging: bool = True,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or None
        if not api_key:
            raise ValueError("OPENAI_API_KEY не установлен")

        self.model = model
        self.temperature = temperature
        self.enable_logging = enable_logging
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        print("Инициализация векторного хранилища...")
        self.vector_store = VectorStore(
            collection_name=collection_name,
            embedding_model=embedding_model,
            api_key=api_key,
            base_url=base_url,
        )

        if self.vector_store.collection.count() == 0:
            print(f"Загрузка документов из {data_file}...")
            self.vector_store.load_documents(data_file)

        print("Инициализация SQLite-кэша...")
        self.cache = RAGCache(db_path=cache_db_path)

        print("Инициализация логгера метрик (SQLite)...")
        self.logger = DatabaseLogger(db_path=logs_db_path, anonymize_users=True)

        print(f"RAG Pipeline инициализирован (OpenAI: {self.model})")

    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Формирование промпта с контекстом из векторной базы."""
        context_parts = [f"Документ {i}:\n{doc['text']}\n" for i, doc in enumerate(context_docs, 1)]
        context = "\n".join(context_parts)

        return f"""Ты - полезный AI ассистент. Ответь на вопрос пользователя на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {query}

Инструкции:
- Отвечай только на основе предоставленного контекста
- Если в контексте нет информации для ответа, честно скажи об этом
- Будь точным и структурированным
- Отвечай на русском языке

Ответ:"""

    def _generate_answer(self, prompt: str) -> str:
        """Генерация ответа через OpenAI Chat Completions API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Ты - полезный AI ассистент, который отвечает на вопросы строго на основе предоставленного контекста.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    def query(
        self,
        user_query: str,
        use_cache: bool = True,
        source: str = "console",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Обрабатывает запрос пользователя, замеряет latency и пишет лог.

        Args:
            user_query: Текст вопроса
            use_cache: Использовать ли кэш ответов
            source: Источник запроса ('console', 'telegram', 'eval')
            user_id: Идентификатор пользователя

        Returns:
            Словарь с ответом, источником и метриками времени
        """
        start_time = time.perf_counter()

        try:
            # Шаг 1: Проверка кэша
            if use_cache:
                cached_result = self.cache.get(user_query)
                if cached_result:
                    response_time_ms = int((time.perf_counter() - start_time) * 1000)
                    if self.enable_logging:
                        self.logger.log_interaction(
                            query=user_query,
                            response=cached_result["answer"],
                            source=source,
                            user_id=user_id,
                            from_cache=True,
                            response_time_ms=response_time_ms,
                            status="success",
                        )
                    return {
                        "query": user_query,
                        "answer": cached_result["answer"],
                        "from_cache": True,
                        "context_docs": cached_result.get("context"),
                        "cached_at": cached_result.get("created_at"),
                        "response_time_ms": response_time_ms,
                    }

            # Шаг 2: Поиск релевантных документов
            context_docs = self.vector_store.search(user_query, top_k=3)

            # Шаг 3: Формирование промпта и генерация ответа
            prompt = self._create_prompt(user_query, context_docs)
            answer = self._generate_answer(prompt)

            # Шаг 4: Сохранение в кэш
            if use_cache:
                context_for_cache = [doc["text"] for doc in context_docs]
                self.cache.set(user_query, answer, context_for_cache)

            response_time_ms = int((time.perf_counter() - start_time) * 1000)

            # Шаг 5: Запись лога
            if self.enable_logging:
                self.logger.log_interaction(
                    query=user_query,
                    response=answer,
                    source=source,
                    user_id=user_id,
                    from_cache=False,
                    response_time_ms=response_time_ms,
                    status="success",
                )

            return {
                "query": user_query,
                "answer": answer,
                "from_cache": False,
                "context_docs": context_docs,
                "model": self.model,
                "response_time_ms": response_time_ms,
            }

        except Exception as e:
            response_time_ms = int((time.perf_counter() - start_time) * 1000)
            if self.enable_logging:
                self.logger.log_interaction(
                    query=user_query,
                    response=f"ERROR: {str(e)}",
                    source=source,
                    user_id=user_id,
                    from_cache=False,
                    response_time_ms=response_time_ms,
                    status="error",
                )
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает объединённую статистику хранилища, кэша и логов."""
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "logs": self.logger.get_stats(),
            "model": self.model,
        }