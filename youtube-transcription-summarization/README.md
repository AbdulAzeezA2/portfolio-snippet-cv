# YouTube Offline Transcription & Summarization Pipeline

A fully offline, privacy-preserving pipeline that downloads YouTube videos, transcribes audio using Whisper, and generates summaries using BART—all running locally without cloud APIs.

## A. Project Overview

This tool provides a complete end-to-end solution for:
- **Downloading** public YouTube videos
- **Extracting** and converting audio with FFmpeg
- **Transcribing** speech to text using offline Whisper models
- **Summarizing** transcripts with offline Transformer models
- **Producing** clean, readable summaries

**Key Features:**
- 100% offline - no cloud APIs required
- Privacy-preserving - all data stays on your machine
- No OpenAI, Google Cloud, or external API dependencies
- Configurable model sizes for speed/accuracy tradeoff
- GPU acceleration support

---

## Requirements

- **Python 3.12+**
- **FFmpeg** (required for audio processing)
- **4GB+ RAM** (8GB+ recommended for larger models)
- **GPU** (optional, for faster processing)

---

## B. Setup & Installation

### 1. Install Python

Download and install Python 3.12 from [python.org/downloads](https://www.python.org/downloads/)

**Important:** During installation, ensure you check:
- Add Python to PATH
- Install pip

### 2. Clone This Repository

```bash
git clone https://github.com/AbdulAzeezA2/portfolio-snippet-cv.git
cd youtube-transcription-summarization
```

### 3. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install FFmpeg

**Windows:**
1. Download FFmpeg from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z)
2. Extract to `C:\ffmpeg\`
3. Ensure `C:\ffmpeg\bin` exists
4. Verify installation:
   ```bash
   ffmpeg -version
   ```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

---

## C. Design Choices and Justification

### Whisper Models (Speech-to-Text)

| Model | Size | Speed | Accuracy | Recommended For |
|-------|------|-------|----------|-----------------|
| `tiny` | ~75MB | 3/3   | 2/6      | Quick tests, low-resource systems |
| `base` | ~150MB | 2.5/3 | 3/6      | Fast transcription, decent quality |
| `small` | ~500MB | 2/3   | 4/6      | **Balanced** (recommended) |
| `medium` | ~1.5GB | 1.5/3 | 5/6      | High accuracy, longer wait |
| `large-v2` | ~3GB | 1/3   | 6/6      | Best quality, requires good hardware |

### Summarization Models

| Model | Description |
|-------|-------------|
| `facebook/bart-large-cnn` | **Default** - Best for coherent summaries |
| `facebook/bart-base` | Faster, lighter alternative |
| `t5-small` | Lightweight, good for testing |

---

## Architecture & Design Decisions

### Why Whisper?
- Fully open-source and offline
- Superior accuracy vs. Vosk, DeepSpeech
- faster-whisper backend for optimized performance
- Multiple model sizes for flexibility

### Why BART?
- Generates highly coherent long-form summaries
- More consistent output than T5
- Faster inference time
- Works well on CPU

### Chunking System
Large transcripts (10,000+ words) exceed transformer token limits (1024-2048 tokens).

**Solution:**
1. Split transcript into 400-word chunks
2. Summarize each chunk independently
3. Combine chunk summaries
4. Generate final summary from combined text

This progressive approach ensures accurate summaries regardless of video length.

---

---

## D. Usage

### Basic Usage

```bash
python yt_summarizer.py -u <YouTube_URL>
```

### Example

```bash
python yt_summarizer.py -u https://www.youtube.com/watch?v=3GMyvX1Jxlc
```

### Advanced Usage

```bash
# Use medium model with GPU acceleration
python yt_summarizer.py -u <URL> -m medium -d cuda

# Custom output directory
python yt_summarizer.py -u <URL> -o my_videos

# Use cookies for restricted videos
python yt_summarizer.py -u <URL> -c cookies.txt

# Different summarization model
python yt_summarizer.py -u <URL> -s facebook/bart-base
```

---

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `-u`, `--url` | **Yes** | — | YouTube video URL |
| `-o`, `--output` | No | `downloads` | Output directory for files |
| `-m`, `--model` | No | `small` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v2` |
| `-d`, `--device` | No | `cpu` | Processing device: `cpu` or `cuda` (GPU) |
| `-s`, `--summary_model` | No | `facebook/bart-large-cnn` | HuggingFace summarization model |
| `-c`, `--cookies` | No | `None` | Path to cookies.txt for restricted videos |
| `--ffmpeg` | No | `C:\ffmpeg\bin` | Path to FFmpeg binaries |

---


## E. Troubleshooting

### Issue: FFmpeg Not Found
**Cause:** FFmpeg not installed or PATH incorrect

**Solution:**
- Ensure FFmpeg is installed
- Use `--ffmpeg` argument to specify path:
  ```bash
  python yt_summarizer.py -u <URL> --ffmpeg C:\ffmpeg\bin
  ```

### Issue: HTTP Error 403 Forbidden
**Cause:** YouTube blocking download requests

**Solution:**
1. Update yt-dlp:
   ```bash
   pip install --upgrade yt-dlp
   ```
2. Use cookies for restricted videos:
   - Export YouTube cookies using browser extension
   - Save as `cookies.txt`
   - Use `-c cookies.txt` argument

### Issue: Out of Memory
**Cause:** Model too large for available RAM

**Solution:**
- Use smaller Whisper model: `-m tiny` or `-m base`
- Close other applications
- Use GPU if available: `-d cuda`

### Issue: Slow Processing
**Cause:** CPU-only inference can be slow

**Solution:**
- Use smaller model: `-m small` or `-m base`
- Enable GPU acceleration: `-d cuda` (requires CUDA-capable GPU)
- Consider upgrading hardware

---

## Output Files

After running, you'll find these files in your output directory:

```
downloads/
├── audio.mp3           # Downloaded audio
├── audio_16k.wav       # Converted audio for Whisper
├── transcript.txt      # Full transcription with timestamps
└── summary.txt         # Final summary
```

---

## Privacy & Security

- **No data leaves your machine** - all processing is local
- **No API keys required** - no risk of key leakage
- **No usage tracking** - your videos, your data
- **Offline capable** - works without internet (after initial model download)

---


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Future Enhancements

- [ ] Support for multiple video formats
- [ ] Batch processing capabilities
- [ ] Web UI interface
- [ ] Speaker diarization
- [ ] Multiple language support
- [ ] Export to different formats (PDF, DOCX)

---