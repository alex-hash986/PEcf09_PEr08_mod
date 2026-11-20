"""
Точка входа для RAG-ассистента (OpenAI API + SQLite Cache + SQLite Logs + Telegram Bot).
"""

import argparse
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline
from telegram_bot import TelegramRAGBot

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def print_banner() -> None:
    banner = """
╔══════════════════════════════════════════════════════════╗
║      RAG Ассистент (OpenAI API + Cache + Logs)          ║
║  Retrieval-Augmented Generation с метриками в SQLite    ║
╚══════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("Команды: 'stats' (статистика), 'logs' (экспорт CSV), 'clear' (очистка кэша), 'exit' (выход)\n")


def print_response(result: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"📝 Вопрос: {result['query']}")
    print(f"{'─'*60}")

    if result["from_cache"]:
        print(f"💾 Источник: КЕШ ({result['response_time_ms']} мс)")
    else:
        print(f"🤖 Источник: {result.get('model', 'OpenAI')} ({result['response_time_ms']} мс)")
        print(f"   Использовано чанков: {len(result.get('context_docs', []))}")

    print(f"\n💬 Ответ:\n{result['answer']}")
    print(f"{'─'*60}\n")


def print_stats(pipeline: RAGPipeline) -> None:
    stats = pipeline.get_stats()
    logs = stats["logs"]

    print(f"\n{'═'*60}")
    print("📊 СТАТИСТИКА И МЕТРИКИ СИСТЕМЫ")
    print(f"{'═'*60}")
    print(f"🗄️  Чанков в ChromaDB: {stats['vector_store']['count']}")
    print(f"💾 Записей в кэше: {stats['cache']['total_entries']} (Hit Rate: {logs['cache_hit_rate']}%)")
    print(f"📝 Всего запросов в логах: {logs['total_requests']} (ошибок: {logs['error_requests']})")
    print(f"👥 Уникальных пользователей: {logs['unique_users']}")
    print(
        f"⏱️  Среднее время: {logs['avg_response_time_ms']} мс "
        f"(кэш: {logs['avg_cached_time_ms']} мс | без кэша: {logs['avg_uncached_time_ms']} мс)"
    )
    print(f"🏷️  Теги запросов: {logs['by_tag']}")
    print(f"{'═'*60}\n")


def run_cli(pipeline: RAGPipeline) -> None:
    print_banner()
    while True:
        try:
            user_input = input("💭 Ваш вопрос: ").strip()
            cmd = user_input.lower()

            if cmd in ["exit", "quit", "q"]:
                print("\n👋 До свидания!")
                break
            if cmd == "stats":
                print_stats(pipeline)
                continue
            if cmd == "logs":
                filename = f"logs_console_{int(time.time())}.csv"
                path = pipeline.logger.export_to_csv(output_path=filename)
                if path:
                    print(f"✅ Логи экспортированы в {path}\n")
                else:
                    print("⚠️  Логи пока пусты\n")
                continue
            if cmd == "clear":
                confirm = input("⚠️  Очистить кэш? (yes/no): ")
                if confirm.lower() in ["yes", "y", "да"]:
                    pipeline.cache.clear()
                    print("✅ Кэш очищен\n")
                continue
            if not user_input:
                continue

            result = pipeline.query(user_input, source="console")
            print_response(result)

        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем.")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-ассистент с метриками и логированием")
    parser.add_argument(
        "--mode",
        choices=["cli", "bot"],
        default=None,
        help="Режим запуска: cli (консоль) или bot (Telegram-бот)",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Ошибка: OPENAI_API_KEY не установлен в .env")
        sys.exit(1)

    pipeline = RAGPipeline(
        collection_name="api_rag_collection",
        cache_db_path="api_rag_cache.db",
        logs_db_path="logs.db",
        data_file="data/docs.txt",
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    mode = args.mode
    if not mode:
        tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if tg_token:
            choice = input("Выберите режим: 1 — Консоль, 2 — Telegram-бот (по умолчанию 1): ").strip()
            mode = "bot" if choice == "2" else "cli"
        else:
            mode = "cli"

    if mode == "bot":
        tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not tg_token:
            print("❌ Ошибка: TELEGRAM_BOT_TOKEN не установлен в .env")
            sys.exit(1)
        bot = TelegramRAGBot(token=tg_token, pipeline=pipeline)
        bot.run()
    else:
        run_cli(pipeline)


if __name__ == "__main__":
    main()