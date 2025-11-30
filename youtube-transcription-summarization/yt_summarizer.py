"""
Author: Abdul Azeez
Date: 30/11/2025
Description:
    This script will do the following things mentioned below:
        1 - Takes a public YouTube video URL as input.
        2 - Download the audio from the video.
        3 - Transcribes the audio into text using an offline speech-to-text model.
        4 - Summarizes the transcribed text using an offline text summarization model.
        5 - Outputs the final summary to the user.

Usage Example:
    python yt_summarizer.py --u https://www.youtube.com/watch?v=3GMyvX1Jxlc
"""

import os
import argparse
import subprocess
from yt_dlp import YoutubeDL
import logging
import warnings
from faster_whisper import WhisperModel
from transformers import BartTokenizer, BartForConditionalGeneration

# Disabled warning for clean printouts, as fixed many things before writing this warning disable cmds
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("moviepy").setLevel(logging.ERROR)


# Step 1: Video and Audio Download
def download_video_audio(url, output_dir, cookiefile=None, ffmpeg_location=None):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, 'audio.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'referer': 'https://www.youtube.com/',
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        "keepvideo": True
    }

    if cookiefile and os.path.exists(cookiefile):
        ydl_opts["cookiefile"] = cookiefile

    if ffmpeg_location:
        ydl_opts["ffmpeg_location"] = ffmpeg_location

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            print(f"Downloaded: {info.get('title', 'video')}")
            return os.path.join(output_dir, 'audio.mp3')
    except Exception as e:
        print(f"Error downloading: {e}")
        return None


# Step 2: Audio conversion to 16 Hz mono wav
def convert_to_16k_mono(input_file, output_file, ffmpeg_location=None):
    ffmpeg_exe = (os.path.join(ffmpeg_location, "ffmpeg.exe") if ffmpeg_location else "ffmpeg")

    cmd = [ ffmpeg_exe, "-y","-i", input_file,"-ar", "16000","-ac", "1","-vn",output_file]

    try:
        subprocess.run(cmd, check=True)
        print(f"Converted to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        raise


# Step 3: Audio transcribe with whisper
def transcribe_whisper(wav_path, model_size, device, out_txt):
    print(f"Loading Whisper model '{model_size}' on {device}...")

    compute_type = "int8" if device == "cpu" else "float16"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print("Transcribing...")
    segments, info = model.transcribe(wav_path, beam_size=5)

    full_text = []
    with open(out_txt, "w", encoding="utf-8") as f:
        for segment in segments:
            line = f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}"
            f.write(line + "\n")
            full_text.append(segment.text)

    print(f"Transcription saved: {out_txt}")
    print(f"Detected language: {info.language}")

    return " ".join(full_text)


# Step 4: Summarize
def summarize_text(text, model_name="facebook/bart-large-cnn", max_length=200):
    """Summarize text using T5 model"""
    print("Loading T5 model for summarization...")
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # T5 needs "summarize: " prefix
    prompt = "summarize: " + text

    # Tokenize with truncation
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)

    summary_ids = model.generate(
        inputs["input_ids"],num_beams=4, max_length=200,min_length=40,length_penalty=2.0,early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def chunk_text(text, max_words=400):
    """Split text into chunks for summarization"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks


# CLI for easy execution
def main():
    parser = argparse.ArgumentParser(
        description="YouTube → Transcription → Summary")

    parser.add_argument("-u", "--url", required=True, help="YouTube video URL")
    parser.add_argument("-o", "--output", default="downloads", help="Output directory")
    parser.add_argument("-m", "--model", default="small", help="Whisper model size")
    parser.add_argument("-d", "--device", default="cpu", help="cpu or cuda")
    parser.add_argument("-s", "--summary_model", default="facebook/bart-large-cnn", help="Summarization model name")
    parser.add_argument("-c", "--cookies", default=None, help="Cookie file path")
    parser.add_argument("--ffmpeg", default=r"C:\ffmpeg\bin", help="FFmpeg folder path")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("YouTube -> Transcription -> Summary")
    print("=" * 60)

    # Step 1: Video and Audio Download
    print("\n[1/4] Downloading video and audio...")
    audio_mp3 = download_video_audio(args.url, args.output, args.cookies, args.ffmpeg)
    if not audio_mp3:
        print("Download failed.")
        return

    # Step 2: Audio conversion to 16 Hz mono wav
    print("\n[2/4] Converting audio...")
    wav_16k = os.path.join(args.output, "audio_16k.wav")
    convert_to_16k_mono(audio_mp3, wav_16k, args.ffmpeg)

    # Step 3: Audio transcribe with whisper
    print("\n[3/4] Transcribing...")
    transcript_file = os.path.join(args.output, "transcript.txt")
    transcript_text = transcribe_whisper(wav_16k, args.model, args.device, transcript_file)

    # Step 4: Summarize
    print("\n[4/4] Summarizing...")

    # If transcript is very long, chunk it
    if len(transcript_text.split()) > 400:
        print("Transcript is long, processing in chunks...")
        chunks = chunk_text(transcript_text, max_words=400)
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  Summarizing chunk {i + 1}/{len(chunks)}...")
            summaries.append(summarize_text(chunk, max_length=150))

        # Combine chunk summaries
        combined = " ".join(summaries)
        if len(combined.split()) > 400:
            final_summary = summarize_text(combined, max_length=200)
        else:
            final_summary = combined
    else:
        final_summary = summarize_text(transcript_text, max_length=200)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(final_summary)
    print("=" * 60)

    summary_file = os.path.join(args.output, "summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print("\nSaved:")
    print("  Transcript:", transcript_file)
    print("  Summary   :", summary_file)
    print("\nDone!")


if __name__ == "__main__":
    main()

