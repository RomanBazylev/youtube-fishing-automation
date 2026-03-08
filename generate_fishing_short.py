import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests
from gtts import gTTS
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    TextClip,
    VideoFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
    vfx,
)

# ── Константы ──────────────────────────────────────────────────────────
TARGET_W, TARGET_H = 1080, 1920
BUILD_DIR = Path("build")
CLIPS_DIR = BUILD_DIR / "clips"
MUSIC_PATH = BUILD_DIR / "music.mp3"

FISHING_TOPICS = [
    "3 ошибки новичков при ловле щуки",
    "Как поймать трофейного окуня — секреты опытных рыбаков",
    "5 приманок, на которые клюёт всегда",
    "Почему у тебя не клюёт? Разбор типичных ошибок",
    "Ночная рыбалка на леща: что нужно знать",
    "Секреты выбора места для рыбалки на реке",
    "Топ-3 проводки на щуку, которые работают в любое время года",
    "Весенний жор: как не упустить свой шанс",
    "Как подобрать снасть для ловли с берега — гайд для новичка",
    "Самая частая причина сходов рыбы и как её устранить",
]

PEXELS_QUERIES = [
    "fishing river",
    "fishing lake",
    "fish underwater",
    "fishing boat",
    "fishing rod",
    "river nature",
    "lake sunrise",
    "angler fishing",
]


@dataclass
class ScriptPart:
    text: str


def ensure_dirs() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)


# ── Фоллбек-сценарий ──────────────────────────────────────────────────
def _fallback_script() -> List[ScriptPart]:
    return [
        ScriptPart("3 ошибки новичков при ловле щуки с лодки!"),
        ScriptPart("Первая — слишком толстая леска. Она убивает чувствительность снасти."),
        ScriptPart("Вторая — слишком быстрая проводка. Щука просто не успевает атаковать."),
        ScriptPart("Третья — ты игнорируешь рельеф. Бровки, коряжник и свалы — вот где трофеи."),
        ScriptPart("Сохрани это видео и подпишись, чтобы ловить больше!"),
    ]


def call_groq_for_script() -> List[ScriptPart]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_script()

    topic = random.choice(FISHING_TOPICS)

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "Ты — топовый сценарист вирусных YouTube Shorts про рыбалку. "
        "Твоя задача — писать сценарии, которые удерживают внимание с первой секунды, "
        "вызывают эмоции и желание поделиться. "
        "Отвечай ТОЛЬКО валидным JSON без markdown-обёрток и пояснений."
    )

    user_prompt = f"""Напиши сценарий YouTube Shorts (30–50 секунд) на тему: «{topic}».

ПРАВИЛА ВИРАЛЬНОСТИ:
- Первая фраза — мощный хук: вопрос, шокирующий факт или провокация. Зритель должен остановить скролл.
- Каждая следующая фраза — короткая (максимум 2 предложения), динамичная, с новой ценностью.
- Используй «ты»-обращение, как будто говоришь с другом на рыбалке.
- Финал — эмоциональный призыв: сохранить, подписаться, написать в комментариях.
- 4–6 частей всего. Язык — живой разговорный русский.

Формат — строго JSON:
{{
  "parts": [
    {{ "text": "..." }}
  ]
}}"""

    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.9,
        "max_tokens": 1024,
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # Убираем markdown-обёртку ```json ... ```, если LLM её добавил
        content = re.sub(r"^```(?:json)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content.strip())
        data = json.loads(content)
        parts = [ScriptPart(p["text"]) for p in data.get("parts", []) if p.get("text")]
        if len(parts) >= 3:
            return parts
    except Exception as exc:
        print(f"[WARN] Groq API error, using fallback script: {exc}")

    return _fallback_script()


# ── Скачивание клипов ─────────────────────────────────────────────────
def _download_file(url: str, dest: Path) -> None:
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


def download_pexels_clips(max_clips: int = 5) -> List[Path]:
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        return []

    query = random.choice(PEXELS_QUERIES)
    headers = {"Authorization": api_key}
    params = {
        "query": query,
        "per_page": max_clips,
        "orientation": "portrait",
    }

    try:
        resp = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"[WARN] Pexels API error: {exc}")
        return []

    data = resp.json()
    result_paths: List[Path] = []

    for idx, video in enumerate(data.get("videos", [])[:max_clips], start=1):
        files = video.get("video_files", [])
        if not files:
            continue
        # Берём HD-качество: ближайший к 1920 по высоте, но не менее 720
        hd_files = [f for f in files if (f.get("height") or 0) >= 720]
        if hd_files:
            best = min(hd_files, key=lambda f: abs((f.get("height") or 0) - 1920))
        else:
            best = max(files, key=lambda f: f.get("height") or 0)
        url = best["link"]
        clip_path = CLIPS_DIR / f"pexels_{idx}.mp4"
        try:
            _download_file(url, clip_path)
            result_paths.append(clip_path)
        except Exception as exc:
            print(f"[WARN] Failed to download Pexels clip {idx}: {exc}")

    return result_paths


def download_pixabay_clips(max_clips: int = 3) -> List[Path]:
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key:
        return []

    params = {
        "key": api_key,
        "q": random.choice(["fishing", "river fish", "lake fishing"]),
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
        print(f"[WARN] Pixabay API error: {exc}")
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
    ]

    for url in random.sample(candidate_urls, len(candidate_urls)):
        try:
            _download_file(url, MUSIC_PATH)
            return MUSIC_PATH
        except Exception:
            continue
    return None


# ── TTS ────────────────────────────────────────────────────────────────
def build_tts_audio(parts: List[ScriptPart]) -> Path:
    text = " ".join(p.text for p in parts)
    tts = gTTS(text=text, lang="ru")
    audio_path = BUILD_DIR / "voice.mp3"
    tts.save(str(audio_path))
    return audio_path


# ── Сборка видео ──────────────────────────────────────────────────────
def _fit_clip_to_frame(clip: VideoFileClip, duration: float) -> VideoFileClip:
    """Подрезает/зацикливает клип до нужной длительности и кропит в 9:16."""
    # Случайное окно по времени
    if clip.duration > duration + 0.5:
        max_start = clip.duration - duration
        start = random.uniform(0, max_start)
        segment = clip.subclip(start, start + duration)
    else:
        segment = clip.fx(vfx.loop, duration=duration)

    # Масштабируем так, чтобы полностью покрыть TARGET_W x TARGET_H, а потом кропим центр
    src_ratio = segment.w / segment.h
    target_ratio = TARGET_W / TARGET_H
    if src_ratio > target_ratio:
        # Видео шире чем нужно — скейлим по высоте, кропим по ширине
        segment = segment.resize(height=TARGET_H)
        x_center = segment.w / 2
        segment = segment.crop(
            x_center=x_center, y_center=TARGET_H / 2,
            width=TARGET_W, height=TARGET_H,
        )
    else:
        # Видео уже чем нужно — скейлим по ширине, кропим по высоте
        segment = segment.resize(width=TARGET_W)
        y_center = segment.h / 2
        segment = segment.crop(
            x_center=TARGET_W / 2, y_center=y_center,
            width=TARGET_W, height=TARGET_H,
        )
    return segment


def _make_subtitle(text: str, duration: float) -> TextClip:
    """Субтитр с обводкой и тенью — читаем даже на ярком фоне."""
    # Фон-подложка для текста (чёрный текст = тень, сдвинутый на 3px)
    shadow = (
        TextClip(
            text,
            fontsize=62,
            color="black",
            font="DejaVu-Sans-Bold",
            method="caption",
            size=(TARGET_W - 100, None),
            stroke_color="black",
            stroke_width=4,
        )
        .set_position(("center", 0.72), relative=True)
        .set_duration(duration)
    )
    # Основной белый текст поверх тени
    main_txt = (
        TextClip(
            text,
            fontsize=62,
            color="white",
            font="DejaVu-Sans-Bold",
            method="caption",
            size=(TARGET_W - 100, None),
            stroke_color="black",
            stroke_width=2,
        )
        .set_position(("center", 0.72), relative=True)
        .set_duration(duration)
    )
    return [shadow, main_txt]


def build_video(
    parts: List[ScriptPart],
    clip_paths: List[Path],
    audio_path: Path,
    music_path: Optional[Path],
) -> Path:
    voice = AudioFileClip(str(audio_path))
    voice_duration = voice.duration

    if not clip_paths:
        raise RuntimeError("No video clips downloaded. Provide PEXELS_API_KEY or PIXABAY_API_KEY.")

    # Длительность каждого кадра пропорционально длине фразы
    raw_durations = []
    for part in parts:
        approx = len(part.text) / 14.0
        raw = min(max(approx, 2.0), 5.0)
        raw_durations.append(raw)

    total_raw = sum(raw_durations) or 1.0
    scale = voice_duration / total_raw
    durations = [d * scale for d in raw_durations]

    # Перемешиваем клипы для разнообразия кадров
    shuffled_clips = clip_paths[:]
    random.shuffle(shuffled_clips)

    source_clips = []  # для корректного закрытия
    video_clips = []
    for i, part in enumerate(parts):
        src_path = shuffled_clips[i % len(shuffled_clips)]
        clip = VideoFileClip(str(src_path))
        source_clips.append(clip)
        target_duration = durations[i]

        fitted = _fit_clip_to_frame(clip, target_duration)

        subtitle_layers = _make_subtitle(part.text, target_duration)

        composed = CompositeVideoClip(
            [fitted] + subtitle_layers,
            size=(TARGET_W, TARGET_H),
        ).set_duration(target_duration)
        video_clips.append(composed)

    video = concatenate_videoclips(video_clips, method="compose").set_duration(voice_duration)

    # Аудио: голос + приглушённая фоновая музыка
    audio_tracks = [voice]
    bg = None
    if music_path and music_path.is_file():
        bg = AudioFileClip(str(music_path)).volumex(0.12)
        bg = bg.set_duration(video.duration)
        audio_tracks.append(bg)

    final_audio = CompositeAudioClip(audio_tracks)
    video = video.set_audio(final_audio).set_duration(voice_duration)

    output_path = BUILD_DIR / "output_fishing_short.mp4"
    video.write_videofile(
        str(output_path),
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="slow",
        bitrate="8000k",
        threads=4,
    )

    # Корректно закрываем все ресурсы
    voice.close()
    if bg is not None:
        bg.close()
    for vc in video_clips:
        vc.close()
    for sc in source_clips:
        sc.close()
    video.close()

    return output_path


def main() -> None:
    ensure_dirs()
    print("[1/5] Generating script...")
    parts = call_groq_for_script()
    print(f"  Script: {len(parts)} parts")
    for i, p in enumerate(parts, 1):
        print(f"  [{i}] {p.text}")

    print("[2/5] Downloading video clips...")
    clip_paths = download_pexels_clips()
    clip_paths += download_pixabay_clips()
    print(f"  Downloaded {len(clip_paths)} clips")

    print("[3/5] Generating TTS audio...")
    audio_path = build_tts_audio(parts)

    print("[4/5] Downloading background music...")
    music_path = download_background_music()

    print("[5/5] Building final video...")
    output = build_video(parts, clip_paths, audio_path, music_path)
    print(f"Done! Video saved to: {output}")


if __name__ == "__main__":
    main()

