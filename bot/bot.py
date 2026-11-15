"""
Telegram-бот поверх API: демонстрация, как модель встроена в продукт.

    BOT_TOKEN=... API_URL=http://localhost:8000 python -m bot.bot

Команды:
  /start, /help                                 — справка
  /example                                      — оценка заявки-примера
  /score <лимит> <возраст> <просрочка> <счёт> <платёж>
      Быстрый режим: история за 6 месяцев считается одинаковой (тот же счёт и платёж каждый месяц),
      просрочка — только в последнем месяце. Это упрощение для демо, API принимает полную заявку.
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx

from app.schemas import EXAMPLE

HELP = (
    "Оцениваю вероятность дефолта по кредитной карте (демо, данные UCI, Тайвань, 2005).\n\n"
    "/score <лимит> <возраст> <просрочка, мес.> <текущий счёт> <последний платёж>\n"
    "например: /score 50000 30 0 20000 2000\n\n"
    "/example — заявка-пример"
)
DECISION = {"approve": "✅ одобрить", "reject": "❌ отказать"}
FEATURE_NAMES = {
    "PAY_1": "статус платежа в последнем месяце", "delay_recent2": "просрочки за 2 последних месяца",
    "delay_max": "максимальная просрочка", "months_since_delay": "давность просрочки", "LIMIT_BAL": "кредитный лимит",
    "util_mean": "загрузка лимита", "pay_ratio_min": "минимальная доля погашения", "delay_count": "число месяцев с просрочкой",
}


class ParseError(ValueError):
    pass


def parse_score_args(text: str) -> dict:
    """«/score 50000 30 0 20000 2000» → заявка в формате API."""
    parts = text.split()[1:]
    if len(parts) != 5:
        raise ParseError("нужно 5 чисел: лимит, возраст, просрочка, счёт, платёж")
    try:
        limit, age, delay, bill, payment = float(parts[0]), int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])
    except ValueError as exc:
        raise ParseError("все параметры должны быть числами") from exc
    if limit <= 0 or not 18 <= age <= 100 or not 0 <= delay <= 9 or payment < 0:
        raise ParseError("лимит > 0, возраст 18–100, просрочка 0–9, платёж ≥ 0")
    app = {"LIMIT_BAL": limit, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 2, "AGE": age}
    app.update({f"PAY_{i}": 0 for i in range(1, 7)})
    app["PAY_1"] = delay
    app.update({f"BILL_AMT{i}": bill for i in range(1, 7)})
    app.update({f"PAY_AMT{i}": payment for i in range(1, 7)})
    return app


def format_prediction(pred: dict) -> str:
    lines = [f"{DECISION[pred['decision']]}", f"PD = {pred['pd']:.1%}, балл {pred['score']}",
             f"порог PD {pred['threshold']:.1%}, модель {pred['model_version']}"]
    if pred["reasons"]:
        lines.append("\nЧто повышает риск:")
        lines += [f"• {FEATURE_NAMES.get(r['feature'], r['feature'])}" for r in pred["reasons"]]
    return "\n".join(lines)


async def score(application: dict, api_url: str) -> dict:
    async with httpx.AsyncClient(base_url=api_url, timeout=10) as client:
        response = await client.post("/predict", json=application)
        response.raise_for_status()
        return response.json()


def build_dispatcher(api_url: str):
    from aiogram import Dispatcher
    from aiogram.filters import Command
    from aiogram.types import Message

    dp = Dispatcher()

    @dp.message(Command("start", "help"))
    async def on_help(message: Message):
        await message.answer(HELP)

    @dp.message(Command("example"))
    async def on_example(message: Message):
        await message.answer(format_prediction(await score(EXAMPLE, api_url)))

    @dp.message(Command("score"))
    async def on_score(message: Message):
        try:
            application = parse_score_args(message.text or "")
        except ParseError as exc:
            await message.answer(f"Не понял: {exc}\n\n{HELP}")
            return
        try:
            await message.answer(format_prediction(await score(application, api_url)))
        except httpx.HTTPError:
            await message.answer("Сервис скоринга недоступен, попробуйте позже.")

    return dp


async def main():
    from aiogram import Bot

    logging.basicConfig(level="INFO")
    bot = Bot(os.environ["BOT_TOKEN"])
    await build_dispatcher(os.getenv("API_URL", "http://localhost:8000")).start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
