## Автоматическая генерация видео о рыбалке для YouTube (GitHub Actions)

Этот проект генерирует **короткие вертикальные ролики (Shorts)** для YouTube‑канала о рыбалке **полностью в GitHub Actions**, без зависимости от локальной машины. Видео может автоматически загружаться на YouTube.

Основной сценарий:

- GitHub Actions по расписанию или вручную запускает workflow `Generate Fishing Short`.
- Скрипт `generate_fishing_short.py`:
  - получает сценарий через LLM (Groq API, модель `llama-3.3-70b-versatile`) с вирусным промптом;
  - LLM сам придумывает уникальную тему (рыба + метод + стиль как контекст);
  - LLM генерирует title, description, tags и поисковые запросы для Pexels;
  - скачивает HD вертикальные клипы с Pexels и Pixabay по LLM-сгенерированным запросам;
  - генерирует озвучку на русском через edge-tts (DmitryNeural, +15% скорость);
  - скачивает бесплатную фоновую музыку;
  - собирает вертикальное видео 1080×1920 с Ken Burns эффектом, плавными переходами, субтитрами с обводкой;
  - загружает видео на YouTube (если настроены credentials).

Итоговый файл `build/output_fishing_short.mp4` и `build/metadata.json` прикладываются как artifacts.

### 1. Настройка репозитория на GitHub

1. Создайте репозиторий и загрузите в него содержимое папки проекта.
2. В разделе `Settings → Secrets and variables → Actions` добавьте секреты:
   - `GROQ_API_KEY` — ключ к Groq LLM API (для генерации сценария);
   - `PEXELS_API_KEY` — ключ к Pexels API (для скачивания видео);
   - `PIXABAY_API_KEY` _(опционально)_ — ключ к Pixabay API (дополнительные клипы).
   - `YOUTUBE_CLIENT_ID` _(опционально)_ — OAuth2 client ID для загрузки на YouTube;
   - `YOUTUBE_CLIENT_SECRET` _(опционально)_ — OAuth2 client secret;
   - `YOUTUBE_REFRESH_TOKEN` _(опционально)_ — OAuth2 refresh token;
   - `YOUTUBE_PRIVACY` _(опционально)_ — `public`, `unlisted` или `private` (по умолчанию `public`).
3. Убедитесь, что в репозитории есть файлы:
   - `generate_fishing_short.py`
   - `upload_youtube.py`
   - `.github/workflows/generate_fishing_short.yml`
   - `requirements.txt`

### 2. GitHub Actions workflow

Workflow описан в `.github/workflows/generate_fishing_short.yml` и делает следующее:

- запускается по расписанию (`cron: 0 18 * * *`) и по кнопке (`workflow_dispatch`);
- ставит Python 3.11, `ffmpeg` и шрифты DejaVu на хосте Ubuntu;
- устанавливает зависимости из `requirements.txt`;
- запускает `python generate_fishing_short.py`;
- если настроены YouTube credentials — загружает видео на канал (`python upload_youtube.py`);
- загружает результат как artifact `fishing-short-{номер_запуска}`.

### 3. Качество видео

Скрипт оптимизирован для максимальной виральности:

- **Разнообразие контента**: каждый запуск выбирает случайную тему и случайные поисковые запросы для стоковых клипов.
- **Вирусный промпт**: LLM генерирует сценарий с хуком в первой секунде, «ты»-обращением и сильным CTA.
- **HD клипы**: загружаются ближайшие к 1920px по высоте, а не минимального качества.
- **Правильный crop**: клипы масштабируются и обрезаются в 9:16 без чёрных полос.
- **Читаемые субтитры**: жирный шрифт 62px, белый текст с чёрной обводкой — читаем на любом фоне.
- **Позиция субтитров**: ~72% от верха экрана — не перекрывают UI YouTube Shorts.
- **Высокий битрейт**: 8 Mbit/s, пресет `medium` — чёткая картинка без артефактов.
- **Ken Burns эффект**: медленный zoom на каждом кадре для кинематографичности.
- **Плавные переходы**: fade-in между клипами.
- **Фоновая музыка**: приглушённая музыка с fadeout в конце.
- **edge-tts озвучка**: мужской голос DmitryNeural, чуть ускоренный для динамики.
- **Метаданные**: LLM генерирует title, description, tags — готовы для YouTube.

### 5. Настройка YouTube (опционально)

Чтобы видео автоматически загружалось на YouTube:

1. Перейдите в [Google Cloud Console](https://console.cloud.google.com/)
2. Создайте проект (или используйте существующий)
3. Включите **YouTube Data API v3** (`APIs & Services → Enable APIs`)
4. Создайте OAuth2 Credentials:
   - `APIs & Services → Credentials → Create Credentials → OAuth client ID`
   - Тип: **Desktop App**
   - Скопируйте `Client ID` и `Client Secret`
5. Получите `refresh_token`:
   - Откройте в браузере (замените `YOUR_CLIENT_ID`):
     ```
     https://accounts.google.com/o/oauth2/auth?client_id=YOUR_CLIENT_ID&redirect_uri=urn:ietf:wg:oauth:2.0:oob&scope=https://www.googleapis.com/auth/youtube.upload&response_type=code&access_type=offline
     ```
   - Авторизуйтесь под аккаунтом YouTube-канала
   - Скопируйте `code` из URL
   - Обменяйте на refresh_token:
     ```bash
     curl -X POST https://oauth2.googleapis.com/token \
       -d "code=YOUR_CODE" \
       -d "client_id=YOUR_CLIENT_ID" \
       -d "client_secret=YOUR_CLIENT_SECRET" \
       -d "redirect_uri=urn:ietf:wg:oauth:2.0:oob" \
       -d "grant_type=authorization_code"
     ```
   - Из ответа скопируйте `refresh_token`
6. Добавьте в GitHub Secrets:
   - `YOUTUBE_CLIENT_ID`
   - `YOUTUBE_CLIENT_SECRET`
   - `YOUTUBE_REFRESH_TOKEN`
   - `YOUTUBE_PRIVACY` (опционально, по умолчанию `public`)

### 6. Локальный тест

1. Установите Python 3.11+ и `ffmpeg` (чтобы он был в `PATH`).
2. В корне проекта выполните:
   ```bash
   pip install -r requirements.txt
   ```
3. Экспортируйте переменные окружения:
   ```bash
   export GROQ_API_KEY=...
   export PEXELS_API_KEY=...
   # опционально:
   export PIXABAY_API_KEY=...
   ```
4. Запустите:
   ```bash
   python generate_fishing_short.py
   ```
5. Готовое видео появится в `build/output_fishing_short.mp4`.

