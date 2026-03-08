import json
import os
import random
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
)


BUILD_DIR = Path("build")
CLIPS_DIR = BUILD_DIR / "clips"
MUSIC_PATH = BUILD_DIR / "music.mp3"


@dataclass
class ScriptPart:
    text: str


def ensure_dirs() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def call_groq_for_script() -> List[ScriptPart]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Фоллбек, если ключа нет – жёстко заданный сценарий
        return [
            ScriptPart("3 ошибки новичков при ловле щуки с лодки."),
            ScriptPart("Первая ошибка — слишком толстая леска, которая убивает чувствительность снасти."),
            ScriptPart("Вторая — слишком быстрая проводка, из-за которой щука не успевает атаковать приманку."),
            ScriptPart("Третья — игнорировать рельеф: бровки, коряжник и свалы часто приносят трофеи."),
            ScriptPart("Если было полезно — подпишись на канал о рыбалке."),
        ]

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    prompt = """
Ты сценарист YouTube-Shorts про рыбалку.
Сгенерируй короткий сценарий (30–60 секунд) на русском языке.

Формат вывода — строго JSON, без лишнего текста:
{
  "parts": [
    { "text": "..." }
  ]
}

Каждый элемент parts — короткая фраза (1–2 предложения), которая будет отдельным кадром с субтитрами.
Тема: советы и ошибки при ловле щуки.
"""
    body = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
    }
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    try:
        data = json.loads(content)
        parts = [ScriptPart(p["text"]) for p in data.get("parts", []) if p.get("text")]
        if parts:
            return parts
    except Exception:
        pass

    # Фоллбек, если LLM вернул невалидный JSON
    return [
        ScriptPart("3 ошибки новичков при ловле щуки с лодки."),
        ScriptPart("Первая ошибка — слишком толстая леска, которая убивает чувствительность снасти."),
        ScriptPart("Вторая — слишком быстрая проводка, из-за которой щука не успевает атаковать приманку."),
        ScriptPart("Третья — игнорировать рельеф: бровки, коряжник и свалы часто приносят трофеи."),
        ScriptPart("Если было полезно — подпишись на канал о рыбалке."),
    ]


def download_pexels_clips(max_clips: int = 3) -> List[Path]:
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        return []

    headers = {"Authorization": api_key}
    params = {
        "query": "fishing",
        "per_page": max_clips,
        "orientation": "portrait",
    }
    resp = requests.get(
        "https://api.pexels.com/videos/search",
        headers=headers,
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    result_paths: List[Path] = []

    for idx, video in enumerate(data.get("videos", [])[:max_clips], start=1):
        files = video.get("video_files", [])
        if not files:
            continue
        # берём файл с минимальной шириной, чтобы не грузить слишком большие
        best = sorted(files, key=lambda f: f.get("width") or 9999)[0]
        url = best["link"]
        clip_path = CLIPS_DIR / f"pexels_{idx}.mp4"
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with clip_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        result_paths.append(clip_path)

    return result_paths


def download_pixabay_clips(max_clips: int = 3) -> List[Path]:
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key:
        return []

    params = {
        "key": api_key,
        "q": "fishing",
        "per_page": max_clips,
        "safesearch": "true",
        "order": "popular",
        # videos API по умолчанию отдаёт несколько размеров; ориентацию подберём по размеру
    }
    resp = requests.get(
        "https://pixabay.com/api/videos/",
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    result_paths: List[Path] = []

    for idx, hit in enumerate(data.get("hits", [])[:max_clips], start=1):
        videos = hit.get("videos") or {}
        # берём medium или small, если medium нет
        cand = videos.get("medium") or videos.get("small") or videos.get("large")
        if not cand or "url" not in cand:
            continue
        url = cand["url"]
        clip_path = CLIPS_DIR / f"pixabay_{idx}.mp4"
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with clip_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        result_paths.append(clip_path)

    return result_paths


def download_background_music() -> Optional[Path]:
    """
    Скачивает один бесплатный трек из заранее заданного списка URL.
    Если что-то пошло не так, просто возвращает None — ролик будет без музыки.
    """
    if os.getenv("DISABLE_BG_MUSIC") == "1":
        return None

    if MUSIC_PATH.is_file():
        return MUSIC_PATH

    # Набор ссылок на бесплатные треки (примерные URL, подставь свои при желании)
    candidate_urls = [
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Komiku/Its_time_for_adventure/Komiku_-_05_-_Friends.mp3",
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Podington_Bear/Daydream/Podington_Bear_-_Daydream.mp3",
    ]

    url = random.choice(candidate_urls)
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with MUSIC_PATH.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return MUSIC_PATH
    except Exception:
        return None


def build_tts_audio(parts: List[ScriptPart]) -> Path:
    text = " ".join(p.text for p in parts)
    tts = gTTS(text=text, lang="ru")
    audio_path = BUILD_DIR / "voice.mp3"
    tts.save(str(audio_path))
    return audio_path


def build_video(
    parts: List[ScriptPart],
    clip_paths: List[Path],
    audio_path: Path,
    music_path: Optional[Path],
) -> Path:
    """
    Собираем вертикальное видео 1080x1920 с субтитрами и звуком:
    - голос (TTS) на нормальном уровне
    - фоновая музыка потише
    """
    voice = AudioFileClip(str(audio_path))
    voice_duration = voice.duration

    # если нет клипов с Pexels — просто один чёрный фон не делаем здесь, завершим ошибкой
    if not clip_paths:
        raise RuntimeError("No Pexels clips downloaded. Provide PEXELS_API_KEY or add local clips.")

    # 1) Даём "сырую" длительность по длине текста (~15 символов/сек),
    #    но держим кадр в комфортных пределах 1.8–3.8 сек.
    raw_durations = []
    for part in parts:
        approx = len(part.text) / 15.0
        raw = min(max(approx, 1.8), 3.8)
        raw_durations.append(raw)

    # 2) Масштабируем так, чтобы сумма длительностей совпадала с длиной озвучки.
    total_raw = sum(raw_durations) or 1.0
    scale = voice_duration / total_raw
    durations = [d * scale for d in raw_durations]

    video_clips = []
    for i, part in enumerate(parts):
        src_path = clip_paths[i % len(clip_paths)]
        clip = VideoFileClip(str(src_path))
        target_duration = durations[i]

        # выбираем случайное окно внутри клипа, чтобы было больше движения
        if clip.duration > target_duration + 0.5:
            max_start = clip.duration - target_duration
            start = random.uniform(0, max_start)
            subclip = clip.subclip(start, start + target_duration)
        else:
            subclip = clip.loop(duration=target_duration)

        # Делаем вертикальное видео 1080x1920 (Full HD вертикальное)
        subclip = subclip.resize(height=1920)

        subtitle = TextClip(
            part.text,
            fontsize=52,
            color="white",
            font="DejaVu-Sans",
            method="caption",
            size=(1080 - 120, None),
        ).set_position(("center", "bottom")).set_duration(target_duration)

        composed = CompositeVideoClip(
            [subclip, subtitle],
            size=(1080, 1920),
        ).set_duration(target_duration)
        video_clips.append(composed)

    video = concatenate_videoclips(video_clips, method="compose").set_duration(voice_duration)

    # Фоновая музыка (если есть)
    audio_tracks = [voice]
    bg = None
    if music_path and music_path.is_file():
        bg = AudioFileClip(str(music_path)).volumex(0.1)
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
        threads=4,
    )

    voice.close()
    if "bg" in locals():
        bg.close()
    for vc in video_clips:
        vc.close()
    video.close()

    return output_path


def main() -> None:
    ensure_dirs()
    parts = call_groq_for_script()
    clip_paths = download_pexels_clips()
    clip_paths += download_pixabay_clips()
    audio_path = build_tts_audio(parts)
    music_path = download_background_music()
    output = build_video(parts, clip_paths, audio_path, music_path)
    print(f"Generated video: {output}")


if __name__ == "__main__":
    main()

