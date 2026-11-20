"""
Модуль для логирования взаимодействий с RAG-ассистентом в SQLite базу данных.
Поддерживает:
- Запись запросов, ответов, времени выполнения и статуса кэша
- Обезличивание идентификаторов пользователей (SHA-256)
- Автоматическое тегирование запросов по тематическим блокам
- Сбор статистики и экспорт в CSV
- Ротацию/очистку старых логов
"""

import csv
import hashlib
import io
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any


class DatabaseLogger:
    """Класс для логирования взаимодействий в SQLite базу данных."""

    def __init__(self, db_path: str = "logs.db", anonymize_users: bool = True):
        """
        Инициализация логгера базы данных.

        Args:
            db_path: Путь к файлу базы данных SQLite
            anonymize_users: Хешировать ли user_id для защиты ПДн
        """
        self.db_path = Path(db_path)
        self.anonymize_users = anonymize_users
        self._init_database()

    def _init_database(self) -> None:
        """Создаёт таблицу и индексы для логов, если они не существуют."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                source TEXT NOT NULL,
                tag TEXT DEFAULT 'general',
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                from_cache INTEGER DEFAULT 0,
                response_time_ms INTEGER,
                status TEXT DEFAULT 'success',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_user_id ON logs(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_source ON logs(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_tag ON logs(tag)")

        conn.commit()
        conn.close()

    def _prepare_user_id(self, user_id: Optional[str]) -> Optional[str]:
        """Обезличивает идентификатор пользователя при включённой настройке."""
        if not user_id:
            return None
        if not self.anonymize_users:
            return str(user_id)
        return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def detect_tag(query: str) -> str:
        """Определяет тематический тег запроса для удобной фильтрации логов."""
        q = query.lower()
        if any(w in q for w in ("rag", "чанк", "chunk", "эмбеддинг", "embedding", "вектор", "chroma")):
            return "rag_architecture"
        if any(w in q for w in ("промпт", "prompt", "few-shot", "chain-of-thought", "температур")):
            return "prompt_engineering"
        if any(w in q for w in ("fine-tuning", "дообучен", "lora", "peft")):
            return "fine_tuning"
        if any(w in q for w in ("обучен", "нейрон", "трансформер", "nlp", "ml")):
            return "ml_basics"
        return "general"

    def log_interaction(
        self,
        query: str,
        response: str,
        source: str = "console",
        user_id: Optional[str] = None,
        from_cache: bool = False,
        response_time_ms: Optional[int] = None,
        status: str = "success",
        tag: Optional[str] = None,
    ) -> None:
        """
        Записывает взаимодействие пользователя с ассистентом в БД.

        Args:
            query: Вопрос пользователя
            response: Ответ ассистента или текст ошибки
            source: Источник запроса (console, telegram)
            user_id: Идентификатор пользователя
            from_cache: Получен ли ответ из кэша
            response_time_ms: Время выполнения запроса в миллисекундах
            status: Статус выполнения ('success' или 'error')
            tag: Тег функционального блока (если None — определяется автоматически)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()
        safe_user_id = self._prepare_user_id(user_id)
        resolved_tag = tag or self.detect_tag(query)

        cursor.execute(
            """
            INSERT INTO logs (
                timestamp, user_id, source, tag, query, response,
                from_cache, response_time_ms, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                safe_user_id,
                source,
                resolved_tag,
                query,
                response,
                1 if from_cache else 0,
                response_time_ms,
                status,
            ),
        )

        conn.commit()
        conn.close()

    def get_logs(
        self,
        limit: Optional[int] = None,
        user_id: Optional[str] = None,
        source: Optional[str] = None,
        tag: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Получает логи из базы данных с поддержкой фильтрации."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        sql = "SELECT * FROM logs WHERE 1=1"
        params: List[Any] = []

        if user_id:
            sql += " AND user_id = ?"
            params.append(self._prepare_user_id(user_id))

        if source:
            sql += " AND source = ?"
            params.append(source)

        if tag:
            sql += " AND tag = ?"
            params.append(tag)

        if start_date:
            sql += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            sql += " AND timestamp <= ?"
            params.append(end_date)

        sql += " ORDER BY timestamp DESC"

        if limit:
            sql += " LIMIT ?"
            params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def export_to_csv(
        self,
        output_path: Optional[str] = None,
        user_id: Optional[str] = None,
        source: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> str:
        """Экспортирует логи в CSV-файл или возвращает CSV-строку."""
        logs = self.get_logs(user_id=user_id, source=source, tag=tag)
        if not logs:
            return ""

        fieldnames = [
            "id",
            "timestamp",
            "user_id",
            "source",
            "tag",
            "query",
            "response",
            "from_cache",
            "response_time_ms",
            "status",
            "created_at",
        ]

        if output_path:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(logs)
            return output_path

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)
        return output.getvalue()

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает агрегированные продуктовые и технические метрики из логов."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM logs")
        total_requests = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM logs WHERE from_cache = 1")
        cached_requests = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM logs WHERE status = 'error'")
        error_requests = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM logs WHERE user_id IS NOT NULL")
        unique_users = cursor.fetchone()[0]

        cursor.execute("SELECT source, COUNT(*) FROM logs GROUP BY source")
        by_source = dict(cursor.fetchall())

        cursor.execute("SELECT tag, COUNT(*) FROM logs GROUP BY tag ORDER BY COUNT(*) DESC")
        by_tag = dict(cursor.fetchall())

        cursor.execute("SELECT AVG(response_time_ms) FROM logs WHERE response_time_ms IS NOT NULL")
        avg_all = cursor.fetchone()[0] or 0

        cursor.execute(
            "SELECT AVG(response_time_ms) FROM logs WHERE from_cache = 1 AND response_time_ms IS NOT NULL"
        )
        avg_cached = cursor.fetchone()[0] or 0

        cursor.execute(
            "SELECT AVG(response_time_ms) FROM logs WHERE from_cache = 0 AND response_time_ms IS NOT NULL"
        )
        avg_uncached = cursor.fetchone()[0] or 0

        conn.close()

        cache_hit_rate = (cached_requests / total_requests * 100) if total_requests else 0.0

        return {
            "total_requests": total_requests,
            "cached_requests": cached_requests,
            "error_requests": error_requests,
            "cache_hit_rate": round(cache_hit_rate, 1),
            "unique_users": unique_users,
            "by_source": by_source,
            "by_tag": by_tag,
            "avg_response_time_ms": round(avg_all, 1),
            "avg_cached_time_ms": round(avg_cached, 1),
            "avg_uncached_time_ms": round(avg_uncached, 1),
        }

    def cleanup_old_logs(self, retention_days: int = 30, max_rows: int = 100000) -> int:
        """
        Ротация и очистка устаревших логов по сроку хранения и лимиту строк.

        Returns:
            Количество удалённых записей
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        cursor.execute("DELETE FROM logs WHERE timestamp < ?", (cutoff,))
        deleted_by_date = cursor.rowcount

        cursor.execute("SELECT COUNT(*) FROM logs")
        current_count = cursor.fetchone()[0]
        deleted_by_limit = 0

        if current_count > max_rows:
            excess = current_count - max_rows
            cursor.execute(
                """
                DELETE FROM logs WHERE id IN (
                    SELECT id FROM logs ORDER BY timestamp ASC LIMIT ?
                )
                """,
                (excess,),
            )
            deleted_by_limit = cursor.rowcount

        conn.commit()
        conn.close()
        return deleted_by_date + deleted_by_limit