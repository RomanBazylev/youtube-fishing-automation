import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
)

# ── Константы ──────────────────────────────────────────────────────────
TARGET_W, TARGET_H = 1080, 1920
BUILD_DIR = Path("build")
CLIPS_DIR = BUILD_DIR / "clips"
AUDIO_DIR = BUILD_DIR / "audio_parts"
MUSIC_PATH = BUILD_DIR / "music.mp3"

# Голос edge-tts: мужской русский, энергичный
TTS_VOICE = "ru-RU-DmitryNeural"
TTS_RATE = "+15%"  # Чуть быстрее для динамики

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
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ── Фоллбек-сценарий ──────────────────────────────────────────────────
def _fallback_script() -> List[ScriptPart]:
    return [
        ScriptPart("3 ошибки новичков при ловле щуки!"),
        ScriptPart("Первая — толстая леска."),
        ScriptPart("Она убивает чувствительность снасти."),
        ScriptPart("Вторая — быстрая проводка."),
        ScriptPart("Щука просто не успевает атаковать."),
        ScriptPart("Третья — ты игнорируешь рельеф."),
        ScriptPart("Бровки, коряжник и свалы — вот где трофеи."),
        ScriptPart("Сохрани и подпишись!"),
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
- Каждая следующая фраза — МАКСИМУМ 1 предложение, 5–15 слов. Короче = динамичнее.
- Используй «ты»-обращение, как будто говоришь с другом на рыбалке.
- Финал — эмоциональный призыв: сохранить, подписаться, написать в комментариях.
- 8–12 частей всего. Каждая часть — отдельный кадр со сменой видео. Чем больше частей, тем динамичнее ролик.
- Язык — живой разговорный русский.

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


def _pexels_best_file(video_files: list) -> Optional[dict]:
    """Pick the best HD file from Pexels video_files list."""
    hd = [f for f in video_files if (f.get("height") or 0) >= 720]
    if hd:
        return min(hd, key=lambda f: abs((f.get("height") or 0) - 1920))
    if video_files:
        return max(video_files, key=lambda f: f.get("height") or 0)
    return None


def download_pexels_clips(target_count: int = 10) -> List[Path]:
    """Download clips using DIFFERENT search queries for visual diversity."""
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        return []

    headers = {"Authorization": api_key}
    queries = random.sample(PEXELS_QUERIES, min(target_count, len(PEXELS_QUERIES)))
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


# ── TTS (edge-tts, по-фразово) ────────────────────────────────────────
async def _generate_part_audio(text: str, out_path: Path) -> None:
    communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
    await communicate.save(str(out_path))


def build_tts_per_part(parts: List[ScriptPart]) -> List[Path]:
    """Генерирует отдельный mp3 для каждой фразы — идеальная синхронизация."""
    audio_paths: List[Path] = []
    for i, part in enumerate(parts):
        out = AUDIO_DIR / f"part_{i}.mp3"
        asyncio.run(_generate_part_audio(part.text, out))
        audio_paths.append(out)
    return audio_paths


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
        from PIL import Image
        import numpy as np
        img = Image.fromarray(frame)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(img)
        # Кропим обратно к исходному размеру из центра
        y_off = (new_h - h) // 2
        x_off = (new_w - w) // 2
        return arr[y_off:y_off + h, x_off:x_off + w]

    return clip.fl(make_frame)


def _make_subtitle(text: str, duration: float) -> list:
    """Субтитр с обводкой — читаем на любом фоне."""
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
    audio_parts: List[Path],
    music_path: Optional[Path],
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

        subtitle_layers = _make_subtitle(part.text, dur)

        composed = CompositeVideoClip(
            [fitted] + subtitle_layers,
            size=(TARGET_W, TARGET_H),
        ).set_duration(dur)
        video_clips.append(composed)

    video = concatenate_videoclips(video_clips, method="compose").set_duration(total_duration)

    # Аудио: голос + приглушённая фоновая музыка
    audio_tracks = [voice]
    bg = None
    if music_path and music_path.is_file():
        bg = AudioFileClip(str(music_path)).volumex(0.12)
        bg = bg.set_duration(total_duration)
        audio_tracks.append(bg)

    final_audio = CompositeAudioClip(audio_tracks)
    video = video.set_audio(final_audio).set_duration(total_duration)

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
    for a in part_audios:
        a.close()
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

    print("[3/5] Generating TTS audio (edge-tts, per-part)...")
    audio_parts = build_tts_per_part(parts)
    for i, ap in enumerate(audio_parts):
        dur = AudioFileClip(str(ap)).duration
        print(f"  Part {i+1}: {dur:.1f}s")

    print("[4/5] Downloading background music...")
    music_path = download_background_music()

    print("[5/5] Building final video...")
    output = build_video(parts, clip_paths, audio_parts, music_path)
    print(f"Done! Video saved to: {output}")


if __name__ == "__main__":
    main()

