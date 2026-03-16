import asyncio
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

import edge_tts
import requests
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    TextClip,
    VideoFileClip,
    CompositeVideoClip,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
    afx,
)

# ── Константы ──────────────────────────────────────────────────────────
TARGET_W, TARGET_H = 1080, 1920
BUILD_DIR = Path("build")
CLIPS_DIR = BUILD_DIR / "clips"
AUDIO_DIR = BUILD_DIR / "audio_parts"
MUSIC_PATH = BUILD_DIR / "music.mp3"
HISTORY_PATH = Path("topic_history.json")
MAX_HISTORY = 20  # remember last N topics to avoid repeats

# Голос edge-tts: ротация для разнообразия
TTS_VOICES = [
    "ru-RU-DmitryNeural",
    "ru-RU-SvetlanaNeural",
]
TTS_RATE_OPTIONS = ["+8%", "+10%", "+12%"]

# Словарь замен для правильного произношения TTS
# edge-tts плохо произносит рыболовные термины (англицизмы, спец. слова)
TTS_PRONUNCIATION_FIXES = {
    "джиг": "джигг",
    "твичинг": "тви́тчинг",
    "ультралайт": "ультра-лайт",
    "троллинг": "тро́ллинг",
    "нахлыст": "на-хлыст",
    "воблер": "во́блер",
    "крэнк": "крэнк",
    "поппер": "по́ппер",
    "балансир": "балан-си́р",
    "фидер": "фи́дер",
    "флюорокарбон": "флюоро-карбон",
    "монофил": "моно-фил",
    "плетёнка": "плетьонка",
    "блесна": "блесна́",
    "мормышка": "мор-мы́шка",
    "дропшот": "дроп-шот",
    "отводной": "от-водно́й",
    "джеркбейт": "джерк-бейт",
    "CTA": "призыв к действию",
}

# Стили подачи — LLM сам выберет тему, а стиль добавит разнообразие
ANGLES = [
    "шок-факт, который удивит даже опытного рыбака",
    "топ ошибок, которые убивают клёв",
    "секретная техника от старого рыбака",
    "мифы vs реальность",
    "лайфхак, который изменит твою рыбалку",
    "честное сравнение: что лучше?",
    "короткая история с неожиданным финалом",
    "научный факт, о котором не знают 90% рыбаков",
    "бюджетный совет, который работает лучше дорогих решений",
    "разбор провальной рыбалки: что пошло не так",
    "гайд для полного новичка за 30 секунд",
    "трофейная тактика, проверенная годами",
]

# Курированные темы по категориям — разнообразный контент
TOPICS = [
    # Советы
    "3 ошибки новичков при ловле щуки",
    "как правильно выбрать воблер для спиннинга",
    "топ 5 приманок для ловли окуня",
    "секреты ловли карпа на фидер",
    "как найти рыбу с помощью эхолота",
    "ночная рыбалка — подготовка и снасти",
    "ловля на живца — правила и секреты",
    "правильная проводка для джига",
    "как привязать крючок — 3 лучших узла",
    "весенняя рыбалка на леща — где искать",
    # Рекорды и истории
    "самая большая щука в истории — 25 кг из Швеции",
    "рекордный сом весом 120 кг из реки По в Италии",
    "рыбак поймал тунца за 3 миллиона долларов",
    "история рекордного марлина на 636 кг",
    "самый большой карп пойманный в России",
    "рыбак боролся с акулой 5 часов — и победил",
    "история уловистого озера Байкал",
    "легенда о щуке Путина весом в 21 кг",
    "золотая рыбка из Англии весом 30 кг",
    "рекордная форель — 22 кг из озера в Канаде",
    # Факты о рыбах
    "рыба-удильщик — самое жуткое создание океана",
    "электрический угорь может убить лошадь",
    "рыба-луна весит до 2300 кг — самая тяжёлая костная рыба",
    "парусник плывёт 110 км/ч — самая быстрая рыба",
    "осётр живёт до 100 лет",
    "пиранья может сожрать корову за 5 минут",
    "рыба-капля — самое грустное животное на Земле",
    "сом может вырасти до 3 метров",
    "камбала рождается обычной, потом глаз переползает",
    "лосось проплывает 3000 км чтобы отложить икру",
    # Лайфхаки
    "как хранить червей чтобы они жили неделю",
    "3 способа быстро почистить рыбу",
    "как сделать прикормку из хлеба и чеснока",
    "зачем натирают приманку WD-40",
    "как поймать раков на сало",
    "секретная наживка из манки для карася",
    "как правильно отпускать рыбу после поимки",
    "лайфхак — использование зубной пасты для блёсен",
    # Экзотические виды
    "арапаима — рыба-гигант из Амазонки до 3 метров",
    "целакант — живое ископаемое возрастом 400 млн лет",
    "латимерия — рыба которую считали вымершей",
    "рыба-меч может пробить лодку",
    "мурена — рыба с двумя челюстями",
    "рыба-камень — самая ядовитая рыба в мире",
    "рыба-попугай ест кораллы и делает песок",
    "гигантский групер может проглотить акулу",
]

PEXELS_QUERIES = [
    "fishing rod water",
    "fisherman river morning",
    "underwater fish swimming",
    "lake sunset fishing",
    "ocean fishing boat",
    "fish caught closeup",
    "fishing net water",
    "river nature scenic",
    "underwater coral reef fish",
    "pike fish predator",
    "fishing reel spinning",
    "boat lake fog morning",
    "carp fish pond",
    "man fishing dock",
    "tropical fish underwater colorful",
    "salmon jumping river",
    "deep sea ocean dark",
    "mountain lake scenic nature",
    "catfish underwater dark",
    "fishing lure bait closeup",
]

PIXABAY_QUERIES = [
    "fishing river nature",
    "underwater fish",
    "lake sunset scenic",
    "fishing boat sea",
    "pike fish predator",
    "coral reef tropical",
    "fisherman rod casting",
    "salmon jumping waterfall",
]


@dataclass
class ScriptPart:
    text: str


@dataclass
class VideoMetadata:
    title: str
    description: str
    tags: List[str]
    topic: str = ""


@dataclass
class WordTiming:
    text: str
    offset: float   # seconds from start
    duration: float  # seconds


# ── Topic deduplication ────────────────────────────────────────────────

def _load_topic_history() -> list:
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_topic_history(history: list) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")


def _pick_unique_topic() -> str:
    """Pick a topic not recently used from curated TOPICS list."""
    history = _load_topic_history()
    available = [t for t in TOPICS if t not in history]
    if not available:
        history = []
        available = list(TOPICS)
    topic = random.choice(available)
    history.append(topic)
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    _save_topic_history(history)
    return topic


def _clean_build_dir() -> None:
    """Remove previous build artifacts to save disk space."""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
        print("  Cleaned previous build directory")


def ensure_dirs() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


FALLBACK_METADATA = VideoMetadata(
    title="Почему щука не клюёт? 3 ошибки новичков 🎣 #shorts",
    description="Узнай 3 главные ошибки при ловле щуки на спиннинг. Сохрани и не повторяй!\n\n#рыбалка #щука #спиннинг #fishing #shorts #лайфхак",
    tags=["рыбалка", "щука", "спиннинг", "fishing", "shorts", "лайфхак"],
)

_CORE_TAGS = ["рыбалка", "fishing", "shorts", "лайфхак", "рыболов"]

_DESCRIPTION_FOOTER = (
    "\n\n#рыбалка #fishing #shorts #лайфхак #рыболов"
    "\nПодпишись — рыбацкие секреты каждый день!"
)


def _enrich_metadata(meta: VideoMetadata) -> VideoMetadata:
    """Ensure title has #shorts, tags have core keywords."""
    title = meta.title
    if "#shorts" not in title.lower():
        title = title.rstrip() + " #shorts"

    tags = list(meta.tags)
    for t in _CORE_TAGS:
        if t not in tags:
            tags.append(t)

    desc = meta.description
    if "#рыбалка" not in desc.lower():
        desc = desc + _DESCRIPTION_FOOTER

    return VideoMetadata(title=title[:100], description=desc, tags=tags)


_FALLBACK_POOL = [
    [
        ScriptPart("Девяносто процентов рыбаков делают эти три ошибки при ловле щуки на спиннинг."),
        ScriptPart("Первая — слишком толстая леска. Ноль тридцать пять и выше убивает всю чувствительность."),
        ScriptPart("Ты просто не чувствуешь поклёвку, и щука выплёвывает приманку до подсечки."),
        ScriptPart("Ставь флюорокарбон ноль двадцать — ноль двадцать пять. В прозрачной воде щука его не видит."),
        ScriptPart("Вторая ошибка — слишком быстрая проводка. Осенью вода ниже десяти градусов."),
        ScriptPart("Щука в холодной воде становится вялой и просто не успевает атаковать."),
        ScriptPart("Делай паузы по две-три секунды между рывками. Именно на паузе щука бьёт в восьмидесяти процентах случаев."),
        ScriptPart("Третья — ты кидаешь куда попало. Щука стоит у бровок, коряжника и перепадов глубины."),
        ScriptPart("Найди на эхолоте свал с трёх на пять метров и кидай точно на границу."),
        ScriptPart("Попробуй эти три правила на ближайшей рыбалке. Напиши в комментариях, сработало ли. Подпишись!"),
    ],
    [
        ScriptPart("Окунь клюёт каждый заброс, если знаешь одну хитрость с проводкой."),
        ScriptPart("Отводной поводок — восемьдесят сантиметров от грузила. Ставь крючок номер два на офсетнике."),
        ScriptPart("Силиконовый твистер три сантиметра, цвет — перламутр или кислотный зелёный."),
        ScriptPart("Бросай под углом сорок пять градусов к берегу. Дай грузилу лечь на дно."),
        ScriptPart("Теперь подтяжка удилищем — пауза две секунды — подмотка слабины."),
        ScriptPart("Окунь атакует на паузе, когда приманка парит над дном. Не торопись."),
        ScriptPart("Лучшее время — утро с шести до девяти, когда малёк активен у берега."),
        ScriptPart("Поклёвка чувствуется как лёгкий тычок. Подсекай резко вверх."),
        ScriptPart("За два часа с этим монтажом можно поймать двадцать-тридцать окуней. Проверено."),
        ScriptPart("Попробуй на ближайшей рыбалке и напиши результат в комментариях. Подпишись!"),
    ],
    [
        ScriptPart("Карп весом больше пяти кило прячется в одних и тех же местах. Вот как их найти."),
        ScriptPart("Ищи бровку — перепад глубины с двух на четыре метра. Маркерным грузом простучи дно."),
        ScriptPart("Корм — кукуруза плюс пеллетс. Замешай с озёрной водой и слепи шары размером с апельсин."),
        ScriptPart("Закорми точку тремя-пятью шарами вечером. Утром карп уже будет кормиться на пятне."),
        ScriptPart("Волосяной монтаж — бойл восемнадцать миллиметров на волосе, крючок номер четыре."),
        ScriptPart("Поводок из мягкой плетёнки, двадцать сантиметров. Грузило — инлайн, шестьдесят грамм."),
        ScriptPart("Заброс точно на прикормленное пятно. Ставь удилище на подставку и жди."),
        ScriptPart("Поклёвка карпа — это резкий загиб сигнализатора. Подсечка плавная, не рвай."),
        ScriptPart("Вываживай мягко, работай фрикционом. Карп делает три-четыре мощных рывка."),
        ScriptPart("Этот метод работает на любом водоёме. Напиши свой рекорд в комментариях!"),
    ],
    [
        ScriptPart("Зимняя рыбалка: три приманки, которые ловят когда всё молчит."),
        ScriptPart("Номер один — мормышка-муравей полтора миллиметра с мотылём. Работает по глухозимью."),
        ScriptPart("Игра — плавные покачивания с паузами каждые пять секунд. Окунь не устоит."),
        ScriptPart("Номер два — балансир пять сантиметров. Цвет — окунёвый или кислотный."),
        ScriptPart("Бросок на дно, потом резкий взмах на тридцать сантиметров и пауза три секунды."),
        ScriptPart("Щука бьёт именно на падении. Следи за кивком."),
        ScriptPart("Номер три — блесна-вертикалка серебро. Классика, которая работает десятилетиями."),
        ScriptPart("Короткие рывки — подброс десять сантиметров, пауза, подброс. Судак обожает."),
        ScriptPart("Главное правило зимой — не сиди на одной лунке дольше десяти минут. Нет поклёвки — перемещайся."),
        ScriptPart("Какая приманка твоя любимая зимой? Пиши в комментариях. Подпишись!"),
    ],
]

_FALLBACK_META_POOL = [
    FALLBACK_METADATA,
    VideoMetadata(
        title="Окунь на каждый заброс! Секрет проводки 🎣 #shorts",
        description="Отводной поводок — убийственный монтаж по окуню. Подробная техника!\n\n#рыбалка #окунь #спиннинг #fishing #shorts",
        tags=["рыбалка", "окунь", "спиннинг", "отводной поводок", "fishing", "shorts"],
    ),
    VideoMetadata(
        title="Как найти карпа 5+ кг? Секрет бровки 🎣 #shorts",
        description="Карп прячется в одних и тех же местах. Вот как их найти!\n\n#рыбалка #карп #фидер #fishing #shorts",
        tags=["рыбалка", "карп", "фидер", "бойлы", "fishing", "shorts"],
    ),
    VideoMetadata(
        title="3 приманки для глухозимья — ловят когда всё молчит 🎣 #shorts",
        description="Зимняя рыбалка когда ничего не клюёт? Эти 3 приманки спасут!\n\n#рыбалка #зимняя #мормышка #балансир #fishing #shorts",
        tags=["рыбалка", "зимняя рыбалка", "мормышка", "балансир", "fishing", "shorts"],
    ),
]


# Фразы-наполнители, которые делают контент слабым
_FILLER_PATTERNS = [
    "мой день начинается", "я не ожидал", "это невероятно", "это было круто",
    "ты не поверишь", "это удивительно", "я был в шоке", "это работает",
    "попробуй сам", "ты должен это знать", "слушай внимательно",
    "сейчас расскажу", "давай разберёмся", "многие не знают",
]


def _validate_script(parts: List[ScriptPart]) -> bool:
    """Проверяет качество сценария. Возвращает True если годный."""
    if len(parts) < 8:
        print(f"[QUALITY] Rejected: too few parts ({len(parts)}, need >=8)")
        return False

    # Средняя длина фразы в словах — минимум 8 слов
    avg_words = sum(len(p.text.split()) for p in parts) / len(parts)
    if avg_words < 7:
        print(f"[QUALITY] Rejected: avg words too low ({avg_words:.1f}, need >=7)")
        return False

    # Проверяем на фразы-наполнители
    filler_count = 0
    for part in parts:
        text_lower = part.text.lower()
        for filler in _FILLER_PATTERNS:
            if filler in text_lower:
                filler_count += 1
                print(f"[QUALITY] Filler detected: '{part.text}'")
                break
    if filler_count > 2:
        print(f"[QUALITY] Rejected: too many fillers ({filler_count})")
        return False

    # Проверяем что хотя бы 60% фраз содержат конкретику (цифры или спец-термины)
    concrete_markers = re.compile(
        r'\d|градус|метр|кило|грамм|миллиметр|секунд|минут|'
        r'блесн|воблер|силикон|леск|крючок|поводок|'
        r'глубин|температур|дно|бровк|коряж|'
        r'ставь|бросай|делай|попробуй|используй|меняй',
        re.IGNORECASE,
    )
    concrete_count = sum(1 for p in parts if concrete_markers.search(p.text))
    ratio = concrete_count / len(parts)
    if ratio < 0.4:
        print(f"[QUALITY] Rejected: not enough concrete content ({ratio:.0%}, need >=40%)")
        return False

    print(f"[QUALITY] Passed: {len(parts)} parts, avg {avg_words:.1f} words, {ratio:.0%} concrete")
    return True


# ── Фоллбек-сценарий ──────────────────────────────────────────────────
def _fallback_script() -> tuple:
    idx = random.randrange(len(_FALLBACK_POOL))
    parts = _FALLBACK_POOL[idx]
    meta = _FALLBACK_META_POOL[idx]
    print(f"[FALLBACK] Using fallback script #{idx + 1}")
    return parts, meta


def call_groq_for_script() -> tuple:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_script()

    topic = _pick_unique_topic()
    angle = random.choice(ANGLES)

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "Ты — опытный рыболов и сценарист вирусных YouTube Shorts. "
        "Ты пишешь сценарии с РЕАЛЬНЫМИ полезными советами для рыбаков. "
        "Каждая фраза должна нести КОНКРЕТНУЮ информацию: цифры, техники, названия приманок, глубины, температуры. "
        "НИКОГДА не пиши пустых фраз-наполнителей вроде 'Мой день начинается рано' или 'Я не ожидал это'. "
        "Каждая фраза = конкретный совет, факт или действие. "
        "Отвечай ТОЛЬКО валидным JSON без markdown-обёрток и пояснений."
    )

    user_prompt = f"""Напиши сценарий YouTube Shorts (45–60 секунд) про рыбалку.

ТЕМА: {topic}
ПОДХОД: {angle}

ТРЕБОВАНИЯ К КОНТЕНТУ:
1. Первая фраза — мощный хук: провокационный вопрос или шокирующий факт с цифрой.
2. КАЖДАЯ фраза должна содержать КОНКРЕТНУЮ пользу: совет, технику, название приманки, размер лески, глубину, время года и т.д.
3. НИКАКИХ пустых фраз-наполнителей. Запрещено: "Я не ожидал это", "Это невероятно", "Мой день начинается рано", "Это было круто".
4. Каждая фраза = 1–2 предложения, 10–25 слов. Достаточно длинно для смысла, но коротко для динамики.
5. Используй «ты»-обращение, как опытный рыбак советует другу.
6. Финал — призыв попробовать совет и написать результат в комментариях.
7. 10–14 частей всего (для 45–60 секунд видео).
8. Язык — живой разговорный русский. Избегай сложных иностранных терминов — если используешь, объясни.

ПРИМЕР ХОРОШЕЙ ФРАЗЫ: "Ставь флюорокарбон ноль двадцать — щука его не видит в прозрачной воде."
ПРИМЕР ПЛОХОЙ: "Это невероятно!" или "Я не ожидал это."

Формат — строго JSON:
{{
  "title": "Цепляющий заголовок для YouTube (до 70 символов) с эмодзи и #shorts",
  "description": "Описание для YouTube (2–3 строки) с хештегами",
  "tags": ["рыбалка", "fishing", "shorts", ...ещё 4–7 тематических тегов],
  "pexels_queries": ["3–5 коротких англ. запросов для поиска видео на Pexels, релевантных теме"],
  "parts": [
    {{ "text": "Фраза с конкретным советом, 10-25 слов" }}
  ]
}}"""

    print(f"  Тема: {topic} | Подход: {angle}")

    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.85,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[WARN] Groq API attempt 1 failed: {exc}, retrying...")
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=45)
            resp.raise_for_status()
        except Exception as exc2:
            print(f"[WARN] Groq API attempt 2 failed: {exc2}, using fallback")
            return _fallback_script()

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        parts = [ScriptPart(p["text"]) for p in data.get("parts", []) if p.get("text")]
        metadata = VideoMetadata(
            title=data.get("title", "")[:100] or "Рыбалка: секреты и лайфхаки #shorts",
            description=data.get("description", "") or "Смотри до конца! #рыбалка #fishing #shorts",
            tags=data.get("tags", ["рыбалка", "fishing", "shorts"]),
            topic=topic,
        )
        metadata = _enrich_metadata(metadata)
        llm_queries = data.get("pexels_queries", [])
        if llm_queries:
            global _llm_pexels_queries
            _llm_pexels_queries = [q for q in llm_queries if isinstance(q, str)][:5]

        if _validate_script(parts):
            return parts, metadata
        print("[WARN] LLM output failed quality check, retrying...")
    except Exception as exc:
        print(f"[WARN] Groq parse error: {exc}, retrying...")

    # ── Retry with reinforced prompt ──
    body["messages"].append({
        "role": "user",
        "content": (
            "ВАЖНО: предыдущий ответ не прошёл проверку качества. "
            "Убедись:\n"
            "1. Минимум 10 фраз, каждая 10-25 слов.\n"
            "2. Каждая фраза содержит конкретику: цифры, названия приманок, глубины.\n"
            "3. НИКАКИХ фраз-наполнителей.\n"
            "Верни JSON в том же формате."
        ),
    })
    body["temperature"] = 1.0
    try:
        resp2 = requests.post(url, headers=headers, json=body, timeout=45)
        resp2.raise_for_status()
        content2 = resp2.json()["choices"][0]["message"]["content"]
        data2 = json.loads(content2)
        parts2 = [ScriptPart(p["text"]) for p in data2.get("parts", []) if p.get("text")]
        metadata2 = VideoMetadata(
            title=data2.get("title", "")[:100] or "Рыбалка: секреты и лайфхаки #shorts",
            description=data2.get("description", "") or "Смотри до конца! #рыбалка #fishing #shorts",
            tags=data2.get("tags", ["рыбалка", "fishing", "shorts"]),
            topic=topic,
        )
        metadata2 = _enrich_metadata(metadata2)
        llm_queries2 = data2.get("pexels_queries", [])
        if llm_queries2:
            _llm_pexels_queries = [q for q in llm_queries2 if isinstance(q, str)][:5]
        if _validate_script(parts2):
            return parts2, metadata2
        print("[WARN] Retry also failed quality check, using fallback")
    except Exception as exc:
        print(f"[WARN] Retry failed: {exc}, using fallback")

    return _fallback_script()


# Глобальная переменная для LLM-сгенерированных запросов Pexels
_llm_pexels_queries: List[str] = []


# ── Скачивание клипов ─────────────────────────────────────────────────
def _download_file(url: str, dest: Path) -> None:
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


def _pexels_best_file(video_files: list) -> Optional[dict]:
    """Pick the best HD file from Pexels video_files list."""
    hd = [f for f in video_files if (f.get("height") or 0) >= 720]
    if hd:
        return min(hd, key=lambda f: abs((f.get("height") or 0) - 1920))
    if video_files:
        return max(video_files, key=lambda f: f.get("height") or 0)
    return None


def download_pexels_clips(target_count: int = 14) -> List[Path]:
    """Download clips using LLM-generated + fallback queries for visual diversity."""
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        return []

    headers = {"Authorization": api_key}
    # Приоритет: LLM-сгенерированные запросы, потом дополняем из дефолтных
    all_queries = list(_llm_pexels_queries)
    extra = [q for q in PEXELS_QUERIES if q not in all_queries]
    random.shuffle(extra)
    all_queries.extend(extra)
    queries = all_queries[:target_count]
    result_paths: List[Path] = []
    seen_ids: set = set()
    clip_idx = 0

    for query in queries:
        if len(result_paths) >= target_count:
            break
        params = {
            "query": query,
            "per_page": 3,
            "orientation": "portrait",
        }
        try:
            resp = requests.get(
                "https://api.pexels.com/videos/search",
                headers=headers, params=params, timeout=30,
            )
            resp.raise_for_status()
        except Exception as exc:
            print(f"[WARN] Pexels search '{query}' failed: {exc}")
            continue

        for video in resp.json().get("videos", []):
            vid_id = video.get("id")
            if vid_id in seen_ids:
                continue
            seen_ids.add(vid_id)
            best = _pexels_best_file(video.get("video_files", []))
            if not best:
                continue
            clip_idx += 1
            clip_path = CLIPS_DIR / f"pexels_{clip_idx}.mp4"
            try:
                _download_file(best["link"], clip_path)
                result_paths.append(clip_path)
                print(f"    Pexels [{query}] -> clip {clip_idx}")
            except Exception as exc:
                print(f"[WARN] Pexels clip {clip_idx} download failed: {exc}")
            if len(result_paths) >= target_count:
                break

    return result_paths


def download_pixabay_clips(max_clips: int = 3) -> List[Path]:
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key:
        return []

    params = {
        "key": api_key,
        "q": random.choice(_llm_pexels_queries or PIXABAY_QUERIES),
        "per_page": max_clips,
        "safesearch": "true",
        "order": "popular",
    }

    try:
        resp = requests.get(
            "https://pixabay.com/api/videos/",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        safe_msg = str(exc)
        if api_key:
            safe_msg = safe_msg.replace(api_key, "***")
        print(f"[WARN] Pixabay API error: {safe_msg}")
        return []

    data = resp.json()
    result_paths: List[Path] = []

    for idx, hit in enumerate(data.get("hits", [])[:max_clips], start=1):
        videos = hit.get("videos") or {}
        cand = videos.get("large") or videos.get("medium") or videos.get("small")
        if not cand or "url" not in cand:
            continue
        url = cand["url"]
        clip_path = CLIPS_DIR / f"pixabay_{idx}.mp4"
        try:
            _download_file(url, clip_path)
            result_paths.append(clip_path)
        except Exception as exc:
            print(f"[WARN] Failed to download Pixabay clip {idx}: {exc}")

    return result_paths


def download_background_music() -> Optional[Path]:
    if os.getenv("DISABLE_BG_MUSIC") == "1":
        return None

    if MUSIC_PATH.is_file():
        return MUSIC_PATH

    candidate_urls = [
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Komiku/Its_time_for_adventure/Komiku_-_05_-_Friends.mp3",
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Podington_Bear/Daydream/Podington_Bear_-_Daydream.mp3",
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/ccCommunity/Chad_Crouch/Arps/Chad_Crouch_-_Shipping_Lanes.mp3",
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Lobo_Loco/Folkish_things/Lobo_Loco_-_01_-_Acoustic_Dreams_ID_1199.mp3",
    ]

    for url in random.sample(candidate_urls, len(candidate_urls)):
        try:
            _download_file(url, MUSIC_PATH)
            return MUSIC_PATH
        except Exception:
            continue
    return None


# ── TTS (edge-tts, по-фразово) ────────────────────────────────────────
def _fix_pronunciation(text: str) -> str:
    """Заменяет сложные для TTS слова на фонетические эквиваленты."""
    result = text
    for word, replacement in TTS_PRONUNCIATION_FIXES.items():
        # Заменяем с учётом регистра
        result = re.sub(re.escape(word), replacement, result, flags=re.IGNORECASE)
    return result


async def _generate_part_audio(
    text: str, voice: str, rate: str, out_path: Path,
) -> List[WordTiming]:
    """Generate audio for one part and capture per-word timings for karaoke."""
    tts_text = _fix_pronunciation(text)
    comm = edge_tts.Communicate(tts_text, voice, rate=rate, boundary="WordBoundary")
    word_timings: List[WordTiming] = []
    audio_chunks = bytearray()

    async for chunk in comm.stream():
        ctype = chunk.get("type") or chunk.get("Type", "")
        if ctype == "audio":
            audio_chunks.extend(chunk["data"])
        elif ctype == "WordBoundary":
            w_text = chunk.get("text") or chunk.get("Text", "")
            w_offset = chunk.get("offset") or chunk.get("Offset", 0)
            w_duration = chunk.get("duration") or chunk.get("Duration", 0)
            if w_text:
                word_timings.append(WordTiming(
                    text=w_text,
                    offset=w_offset / 10_000_000,
                    duration=w_duration / 10_000_000,
                ))

    with out_path.open("wb") as f:
        f.write(audio_chunks)
    return word_timings


async def _generate_all_audio(parts: List[ScriptPart]) -> tuple:
    """Генерирует все аудио-фразы последовательно, собирает word timings."""
    voice = random.choice(TTS_VOICES)
    rate = random.choice(TTS_RATE_OPTIONS)
    print(f"  TTS voice: {voice}, rate: {rate}")
    audio_paths: List[Path] = []
    all_timings: List[List[WordTiming]] = []

    for i, part in enumerate(parts):
        out = AUDIO_DIR / f"part_{i}.mp3"
        audio_paths.append(out)
        timings = await _generate_part_audio(part.text, voice, rate, out)
        all_timings.append(timings)

    return audio_paths, all_timings


def build_tts_per_part(parts: List[ScriptPart]) -> tuple:
    """Генерирует отдельный mp3 для каждой фразы + word timings для караоке."""
    return asyncio.run(_generate_all_audio(parts))


# ── Сборка видео ──────────────────────────────────────────────────────
def _fit_clip_to_frame(clip: VideoFileClip, duration: float) -> VideoFileClip:
    """Подрезает/зацикливает клип до нужной длительности и кропит в 9:16."""
    if clip.duration > duration + 0.5:
        max_start = clip.duration - duration
        start = random.uniform(0, max_start)
        segment = clip.subclip(start, start + duration)
    else:
        segment = clip.fx(vfx.loop, duration=duration)

    # Масштабируем с запасом +10% для Ken Burns, потом кропим центр
    margin = 1.10
    src_ratio = segment.w / segment.h
    target_ratio = TARGET_W / TARGET_H
    if src_ratio > target_ratio:
        segment = segment.resize(height=int(TARGET_H * margin))
    else:
        segment = segment.resize(width=int(TARGET_W * margin))

    # Кропим точно в 9:16
    segment = segment.crop(
        x_center=segment.w / 2, y_center=segment.h / 2,
        width=TARGET_W, height=TARGET_H,
    )
    return segment


def _apply_ken_burns(clip, duration: float):
    """Медленный zoom-in или zoom-out для динамики кадра."""
    direction = random.choice(["in", "out"])
    start_scale = 1.0
    end_scale = random.uniform(1.06, 1.12)
    if direction == "out":
        start_scale, end_scale = end_scale, start_scale

    def make_frame(get_frame, t):
        progress = t / max(duration, 0.01)
        scale = start_scale + (end_scale - start_scale) * progress
        frame = get_frame(t)
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img = Image.fromarray(frame)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(img)
        # Кропим обратно к исходному размеру из центра
        y_off = (new_h - h) // 2
        x_off = (new_w - w) // 2
        return arr[y_off:y_off + h, x_off:x_off + w]

    return clip.fl(make_frame)


def _make_karaoke_subtitle(
    word_timings: List[WordTiming], duration: float, is_hook: bool = False,
) -> list:
    """Караоке-субтитры: подсветка по словам (оранжевый/жёлтый → белый)."""
    if not word_timings:
        return []

    CHUNK_SIZE = 3
    chunks = []
    for i in range(0, len(word_timings), CHUNK_SIZE):
        chunks.append(word_timings[i:i + CHUNK_SIZE])

    active_color = "#FF6600" if is_hook else "yellow"
    active_fontsize = 80 if is_hook else 68
    fade_fontsize = 72 if is_hook else 68

    layers = []
    for ci, chunk in enumerate(chunks):
        chunk_start = chunk[0].offset
        if ci + 1 < len(chunks):
            chunk_end = chunks[ci + 1][0].offset
        else:
            chunk_end = min(chunk[-1].offset + chunk[-1].duration + 0.3, duration)
        chunk_dur = chunk_end - chunk_start
        if chunk_dur <= 0:
            continue

        full_text = " ".join(w.text for w in chunk)
        speak_end = chunk[-1].offset + chunk[-1].duration
        speak_dur = speak_end - chunk_start

        if speak_dur > 0:
            active_txt = (
                TextClip(
                    full_text,
                    fontsize=active_fontsize,
                    color=active_color,
                    font="DejaVu-Sans-Bold",
                    method="caption",
                    size=(TARGET_W - 100, None),
                    stroke_color="black",
                    stroke_width=4,
                )
                .set_position(("center", 0.75), relative=True)
                .set_start(chunk_start)
                .set_duration(min(speak_dur, chunk_dur))
            )
            layers.append(active_txt)

        remaining = chunk_end - speak_end
        if remaining > 0.05:
            white_txt = (
                TextClip(
                    full_text,
                    fontsize=fade_fontsize,
                    color="white",
                    font="DejaVu-Sans-Bold",
                    method="caption",
                    size=(TARGET_W - 100, None),
                    stroke_color="black",
                    stroke_width=3,
                )
                .set_position(("center", 0.75), relative=True)
                .set_start(speak_end)
                .set_duration(remaining)
            )
            layers.append(white_txt)

    return layers


def build_video(
    parts: List[ScriptPart],
    clip_paths: List[Path],
    audio_parts: List[Path],
    music_path: Optional[Path],
    word_timings: List[List[WordTiming]],
) -> Path:
    if not clip_paths:
        raise RuntimeError("No video clips downloaded. Provide PEXELS_API_KEY or PIXABAY_API_KEY.")

    # Загружаем аудио каждой фразы — длительность точная
    part_audios = [AudioFileClip(str(p)) for p in audio_parts]
    durations = [a.duration for a in part_audios]
    total_duration = sum(durations)

    # Объединяем аудио-фразы в один трек
    voice = concatenate_audioclips(part_audios)

    # Гарантируем что каждый кадр берёт уникальный клип (если клипов хватает)
    if len(clip_paths) >= len(parts):
        chosen_clips = random.sample(clip_paths, len(parts))
    else:
        chosen_clips = clip_paths[:]
        random.shuffle(chosen_clips)
        while len(chosen_clips) < len(parts):
            chosen_clips.append(random.choice(clip_paths))

    source_clips = []
    video_clips = []
    for i, part in enumerate(parts):
        src_path = chosen_clips[i]
        clip = VideoFileClip(str(src_path))
        source_clips.append(clip)
        dur = durations[i]

        fitted = _fit_clip_to_frame(clip, dur)
        fitted = _apply_ken_burns(fitted, dur)

        timings = word_timings[i] if i < len(word_timings) else []
        subtitle_layers = _make_karaoke_subtitle(timings, dur, is_hook=(i == 0))

        composed = CompositeVideoClip(
            [fitted] + subtitle_layers,
            size=(TARGET_W, TARGET_H),
        ).set_duration(dur)
        video_clips.append(composed)

    # Плавный fade-in (0.2 сек) для каждого клипа кроме первого
    # НЕ используем crossfade с negative padding — он ломает синхронизацию аудио/субтитров
    FADE_DUR = 0.2
    for idx in range(1, len(video_clips)):
        video_clips[idx] = video_clips[idx].crossfadein(FADE_DUR)

    video = concatenate_videoclips(video_clips, method="compose").set_duration(total_duration)

    # Аудио: голос + приглушённая фоновая музыка
    audio_tracks = [voice]
    bg = None
    if music_path and music_path.is_file():
        bg = AudioFileClip(str(music_path)).volumex(0.12)
        bg = bg.set_duration(total_duration)
        # Плавное затухание музыки в конце
        bg = bg.fx(afx.audio_fadeout, min(1.5, total_duration * 0.1))
        audio_tracks.append(bg)

    final_audio = CompositeAudioClip(audio_tracks)
    video = video.set_audio(final_audio).set_duration(total_duration)

    output_path = BUILD_DIR / "output_fishing_short.mp4"
    video.write_videofile(
        str(output_path),
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        bitrate="8000k",
        threads=4,
    )

    # Корректно закрываем все ресурсы
    voice.close()
    if bg is not None:
        bg.close()
    for a in part_audios:
        a.close()
    for vc in video_clips:
        vc.close()
    for sc in source_clips:
        sc.close()
    video.close()

    return output_path


def _save_metadata(meta: VideoMetadata) -> None:
    """Сохраняет метаданные видео в JSON для будущей автозагрузки."""
    meta_path = BUILD_DIR / "metadata.json"
    meta_path.write_text(
        json.dumps(
            {"title": meta.title, "description": meta.description, "tags": meta.tags, "topic": meta.topic},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Metadata saved to {meta_path}")


def main() -> None:
    _clean_build_dir()
    ensure_dirs()
    print("[1/5] Generating script...")
    parts, metadata = call_groq_for_script()
    print(f"  Script: {len(parts)} parts")
    print(f"  Title: {metadata.title}")
    total_words = 0
    for i, p in enumerate(parts, 1):
        wc = len(p.text.split())
        total_words += wc
        print(f"  [{i}] ({wc}w) {p.text}")
    est_duration = total_words / 2.3  # ~2.3 слова/сек для русского TTS
    print(f"  Estimated duration: ~{est_duration:.0f}s ({total_words} words)")
    _save_metadata(metadata)

    print("[2/5] Downloading video clips...")
    clip_paths = download_pexels_clips()
    clip_paths += download_pixabay_clips()
    print(f"  Downloaded {len(clip_paths)} clips")

    print("[3/5] Generating TTS audio (edge-tts, per-part)...")
    audio_parts, word_timings = build_tts_per_part(parts)
    for i, ap in enumerate(audio_parts):
        a = AudioFileClip(str(ap))
        print(f"  Part {i+1}: {a.duration:.1f}s")
        a.close()

    print("[4/5] Downloading background music...")
    music_path = download_background_music()

    print("[5/5] Building final video...")
    output = build_video(parts, clip_paths, audio_parts, music_path, word_timings)
    print(f"Done! Video saved to: {output}")


if __name__ == "__main__":
    main()

