"""
Telegram-бот для RAG-ассистента с логированием взаимодействий и метриками.
"""

import os
import time
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from rag_pipeline import RAGPipeline


class TelegramRAGBot:
    """Telegram-бот для взаимодействия с RAGPipeline."""

    def __init__(self, token: str, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.application = Application.builder().token(token).build()

        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("logs", self.logs_command))
        self.application.add_handler(CommandHandler("cleanup", self.cleanup_command))

        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        welcome_text = (
            "🤖 *RAG-ассистент по базе знаний*\n\n"
            "Задайте вопрос по машинному обучению, RAG, векторным базам данных или промпт-инжинирингу.\n\n"
            "*Доступные команды:*\n"
            "/help — справка и примеры вопросов\n"
            "/stats — метрики системы (кэш, latency, логи)\n"
            "/logs — выгрузить историю ваших запросов в CSV\n"
            "/cleanup — очистить устаревшие логи (>30 дней)"
        )
        await update.message.reply_text(welcome_text, parse_mode="Markdown")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        help_text = (
            "📚 *Как работает ассистент:*\n"
            "1. Проверяет SQLite-кэш (SHA-256) для мгновенного ответа.\n"
            "2. Ищет топ-3 релевантных чанка в ChromaDB.\n"
            "3. Формирует ответ через OpenAI API строго по контексту.\n"
            "4. Фиксирует метрики (latency, тег, статус кэша) в `logs.db`.\n\n"
            "*Примеры вопросов:*\n"
            "• Что такое RAG и как он работает?\n"
            "• Какие бывают стратегии chunking?\n"
            "• Зачем нужна температура в языковых моделях?"
        )
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            stats = self.pipeline.get_stats()
            vs = stats["vector_store"]
            cache = stats["cache"]
            logs = stats["logs"]

            tags_lines = "\n".join(
                [f"  • `{tag}`: {cnt}" for tag, cnt in logs["by_tag"].items()]
            ) or "  • пока нет данных"

            stats_text = (
                "📊 *МЕТРИКИ И СТАТИСТИКА СИСТЕМЫ*\n\n"
                f"🗄 *Векторная база (ChromaDB):*\n"
                f"  • Чанков в индексе: {vs['count']}\n"
                f"  • Модель LLM: `{stats['model']}`\n\n"
                f"💾 *Кэш ответов (SQLite):*\n"
                f"  • Записей в кэше: {cache['total_entries']}\n"
                f"  • Cache Hit Rate: *{logs['cache_hit_rate']}%*\n\n"
                f"📝 *Логи взаимодействий (`logs.db`):*\n"
                f"  • Всего запросов: {logs['total_requests']}\n"
                f"  • Из кэша / новых: {logs['cached_requests']} / {logs['total_requests'] - logs['cached_requests']}\n"
                f"  • Ошибок: {logs['error_requests']}\n"
                f"  • Уникальных пользователей: {logs['unique_users']}\n\n"
                f"⏱ *Скорость ответа (Latency):*\n"
                f"  • Среднее общее: *{logs['avg_response_time_ms']:.0f} мс*\n"
                f"  • Из кэша: *{logs['avg_cached_time_ms']:.0f} мс*\n"
                f"  • Без кэша (RAG + LLM): *{logs['avg_uncached_time_ms']:.0f} мс*\n\n"
                f"🏷 *Распределение по тегам:*\n{tags_lines}"
            )
            await update.message.reply_text(stats_text, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка получения статистики: {e}")

    async def logs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            user_id = str(update.effective_user.id)
            csv_content = self.pipeline.logger.export_to_csv(user_id=user_id)

            if not csv_content:
                await update.message.reply_text("📝 Для вашего аккаунта пока нет записей в логах.")
                return

            filename = f"rag_logs_{int(time.time())}.csv"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(csv_content)

            try:
                with open(filename, "rb") as f:
                    await update.message.reply_document(
                        document=f,
                        filename=filename,
                        caption="📊 Экспорт ваших логов взаимодействий (обезличенный user_id)",
                    )
            finally:
                if os.path.exists(filename):
                    os.remove(filename)

        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка экспорта логов: {e}")

    async def cleanup_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            deleted = self.pipeline.logger.cleanup_old_logs(retention_days=30, max_rows=100000)
            await update.message.reply_text(
                f"🧹 Ротация логов выполнена. Удалено устаревших записей: {deleted}."
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка очистки логов: {e}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_message = update.message.text.strip()
        user_id = str(update.effective_user.id)

        await update.message.chat.send_action(action="typing")

        try:
            result = self.pipeline.query(
                user_query=user_message,
                use_cache=True,
                source="telegram",
                user_id=user_id,
            )

            answer = result["answer"]
            max_length = 4000
            for i in range(0, len(answer), max_length):
                await update.message.reply_text(answer[i : i + max_length])

            source_badge = "💾 из кэша" if result["from_cache"] else f"🤖 {result.get('model', 'OpenAI')}"
            await update.message.reply_text(
                f"_{source_badge} • {result['response_time_ms']} мс_",
                parse_mode="Markdown",
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка при обработке запроса: {e}")

    def run(self) -> None:
        print("🤖 Запуск Telegram-бота RAG-ассистента...")
        self.application.run_polling()