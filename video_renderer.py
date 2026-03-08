import argparse
import json
import random
from pathlib import Path

from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build vertical Shorts-style video from frames JSON and TTS audio.",
    )
    parser.add_argument(
        "--frames-json",
        required=True,
        help="Path to JSON file with frames description (see docs/media_and_tts.md).",
    )
    parser.add_argument(
        "--audio-path",
        required=True,
        help="Path to TTS audio file (mp3/wav).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the resulting MP4 will be written.",
    )
    parser.add_argument(
        "--assets-root",
        default="assets/images",
        help="Root directory with image assets grouped by category.",
    )
    parser.add_argument(
        "--resolution",
        default="1080x1920",
        help="Target video resolution in WIDTHxHEIGHT format (default: 1080x1920).",
    )
    return parser.parse_args()


def load_frames(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("frames JSON must be a list")
    return data


def pick_asset_for_category(assets_root: Path, category: str) -> Path:
    category_dir = assets_root / category
    if not category_dir.is_dir():
        raise FileNotFoundError(f"Category directory not found: {category_dir}")

    candidates = [
        p
        for p in category_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not candidates:
        raise FileNotFoundError(f"No image files found in {category_dir}")

    return random.choice(candidates)


def parse_resolution(resolution: str) -> tuple[int, int]:
    try:
        w_str, h_str = resolution.lower().split("x", 1)
        return int(w_str), int(h_str)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid resolution format: {resolution}") from exc


def build_video(
    frames: list[dict],
    audio_path: Path,
    assets_root: Path,
    resolution: str,
    output_dir: Path,
) -> Path:
    width, height = parse_resolution(resolution)

    audio_clip = AudioFileClip(str(audio_path))
    audio_duration = audio_clip.duration

    base_durations = [max(float(frame.get("duration", 3.0)), 0.5) for frame in frames]
    total_base = sum(base_durations) or 1.0
    scale = audio_duration / total_base

    scaled_durations = [d * scale for d in base_durations]

    video_clips: list[ImageClip] = []
    for frame, duration in zip(frames, scaled_durations, strict=True):
        category = str(frame.get("category", "river"))
        asset_path = pick_asset_for_category(assets_root, category)

        img_clip = (
            ImageClip(str(asset_path))
            .set_duration(duration)
            .resize(height=height)
            .set_position("center")
        )
        video_clips.append(img_clip)

    video = concatenate_videoclips(video_clips, method="compose")
    video = video.set_audio(audio_clip).set_duration(audio_duration)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output_shorts.mp4"

    video.write_videofile(
        str(output_path),
        fps=30,
        codec="libx264",
        audio_codec="aac",
        threads=4,
    )

    audio_clip.close()
    for clip in video_clips:
        clip.close()
    video.close()

    return output_path


def main() -> None:
    args = parse_args()

    frames_path = Path(args.frames_json)
    audio_path = Path(args.audio_path)
    output_dir = Path(args.output_dir)
    assets_root = Path(args.assets_root)

    frames = load_frames(frames_path)
    output_path = build_video(
        frames=frames,
        audio_path=audio_path,
        assets_root=assets_root,
        resolution=args.resolution,
        output_dir=output_dir,
    )

    print(f"Video written to: {output_path}")


if __name__ == "__main__":
    main()

