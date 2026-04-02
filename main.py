#!/usr/bin/env python3
"""
🎙️ Audio Dubbing Bot — Ultra Version
Powered by Groq Whisper + Groq LLaMA + Edge TTS
Python 3.11 | PTB 20.7
"""

import os, io, time, asyncio, logging, threading, tempfile, functools
import requests
from concurrent.futures import ThreadPoolExecutor
from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)
from groq import Groq

# ── FFmpeg fix for Render (apt-get কাজ না করলেও এটা কাজ করে) ──
import imageio_ffmpeg
from pydub import AudioSegment
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.ffprobe   = imageio_ffmpeg.get_ffmpeg_exe()

from gtts import gTTS

# ══════════════════════════════════════════════
# ⚙️  CONFIG
# ══════════════════════════════════════════════
BOT_TOKEN    = os.environ.get('BOT_TOKEN', '')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
RENDER_URL   = os.environ.get('RENDER_URL', '')

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── Multiple Groq Keys (GROQ_API_KEY_1, _2, ... auto-load)
_groq_keys = []
if os.environ.get('GROQ_API_KEY', ''):
    _groq_keys.append(os.environ['GROQ_API_KEY'])
for _i in range(1, 10):
    _k = os.environ.get(f'GROQ_API_KEY_{_i}', '')
    if _k and _k not in _groq_keys:
        _groq_keys.append(_k)
if not _groq_keys:
    _groq_keys = ['dummy_key']

executor = ThreadPoolExecutor(max_workers=8)

# ══════════════════════════════════════════════
# 🔄  GROQ KEY MANAGER
# ══════════════════════════════════════════════
class GroqManager:
    def __init__(self, keys):
        self._clients = [Groq(api_key=k) for k in keys]
        self._idx     = 0
        self._lock    = threading.Lock()
        logger.info(f"✅ Groq keys loaded: {len(self._clients)}টি")

    def client(self):
        return self._clients[self._idx]

    def rotate(self):
        with self._lock:
            self._idx = (self._idx + 1) % len(self._clients)
            logger.warning(f"🔄 Groq key #{self._idx + 1}-এ switch")
        return self._clients[self._idx]

    def chat(self, **kwargs):
        tried = 0
        while tried < len(self._clients):
            try:
                return self.client().chat.completions.create(**kwargs)
            except Exception as e:
                es = str(e).lower()
                if any(x in es for x in ['quota', 'limit exceeded', '402', 'billing', 'rate']):
                    self.rotate(); tried += 1
                    if tried >= len(self._clients):
                        raise Exception("❌ সব Groq API key-এর limit শেষ!")
                    time.sleep(1)
                    continue
                raise
        raise Exception("❌ Groq API error!")

    def transcribe(self, **kwargs):
        tried = 0
        while tried < len(self._clients):
            try:
                return self.client().audio.transcriptions.create(**kwargs)
            except Exception as e:
                es = str(e).lower()
                if any(x in es for x in ['quota', 'limit exceeded', '402', 'billing', 'rate']):
                    self.rotate(); tried += 1
                    if tried >= len(self._clients):
                        raise Exception("❌ সব Groq API key-এর limit শেষ!")
                    time.sleep(1)
                    continue
                raise
        raise Exception("❌ Groq Whisper error!")

groq_mgr = GroqManager(_groq_keys)

# ══════════════════════════════════════════════
# 🌐  LANGUAGE CONFIG
# ══════════════════════════════════════════════
LANGUAGES = {
    'bn': '🇧🇩 বাংলা',
    'en': '🇺🇸 English',
    'hi': '🇮🇳 Hindi',
    'ar': '🇸🇦 Arabic',
    'fr': '🇫🇷 French',
    'es': '🇪🇸 Spanish',
    'de': '🇩🇪 German',
    'ja': '🇯🇵 Japanese',
    'ko': '🇰🇷 Korean',
    'zh': '🇨🇳 Chinese',
    'ru': '🇷🇺 Russian',
    'tr': '🇹🇷 Turkish',
    'it': '🇮🇹 Italian',
    'ur': '🇵🇰 Urdu',
}

TTS_VOICES = {
    'bn': 'bn',
    'en': 'en',
    'hi': 'hi',
    'ar': 'ar',
    'fr': 'fr',
    'es': 'es',
    'de': 'de',
    'ja': 'ja',
    'ko': 'ko',
    'zh': 'zh-CN',
    'ru': 'ru',
    'tr': 'tr',
    'it': 'it',
    'ur': 'ur',
}

LANG_FULL_NAMES = {
    'bn': 'Bengali (বাংলা)',
    'en': 'English',
    'hi': 'Hindi (हिन्दी)',
    'ar': 'Arabic (العربية)',
    'fr': 'French (Français)',
    'es': 'Spanish (Español)',
    'de': 'German (Deutsch)',
    'ja': 'Japanese (日本語)',
    'ko': 'Korean (한국어)',
    'zh': 'Chinese (中文)',
    'ru': 'Russian (Русский)',
    'tr': 'Turkish (Türkçe)',
    'it': 'Italian (Italiano)',
    'ur': 'Urdu (اردو)',
}

AUDIO_EXTENSIONS = ('.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac', '.opus', '.wma', '.webm')

# ══════════════════════════════════════════════
# 📦  STATE MANAGEMENT
# ══════════════════════════════════════════════
user_audio = {}   # {uid: {file_id, file_name, duration, size}}
processing = {}   # {uid: True}

# ══════════════════════════════════════════════
# 🌐  FLASK (Keep Alive for Render)
# ══════════════════════════════════════════════
flask_app = Flask(__name__)

@flask_app.route('/')
def home():
    return '🎙️ Audio Dubbing Bot is Running! ✅'

@flask_app.route('/health')
def health():
    return {'status': 'ok', 'bot': 'Audio Dubbing Bot', 'time': time.time()}

def run_flask():
    port = int(os.environ.get('PORT', 10000))
    flask_app.run(host='0.0.0.0', port=port, debug=False)

def self_ping():
    """Render free tier sleep প্রতিরোধে নিজেকে ping করে"""
    if not RENDER_URL:
        logger.info("⚠️ RENDER_URL নেই, self-ping বন্ধ")
        return
    while True:
        try:
            r = requests.get(RENDER_URL, timeout=15)
            logger.info(f"✅ Self-ping OK [{r.status_code}]")
        except Exception as e:
            logger.warning(f"⚠️ Self-ping failed: {e}")
        time.sleep(840)  # 14 মিনিট পরপর ping

# ══════════════════════════════════════════════
# 🎨  KEYBOARDS
# ══════════════════════════════════════════════
def get_source_language_keyboard():
    """মূল অডিওর ভাষা সিলেক্ট করার keyboard"""
    lang_items = list(LANGUAGES.items())
    keyboard = []
    for i in range(0, len(lang_items), 2):
        row = []
        for code, name in lang_items[i:i+2]:
            row.append(InlineKeyboardButton(name, callback_data=f"src_{code}"))
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("❌ বাতিল করো", callback_data="dub_cancel")])
    return InlineKeyboardMarkup(keyboard)

def get_language_keyboard():
    """ডাবিং ভাষা সিলেক্ট করার keyboard"""
    lang_items = list(LANGUAGES.items())
    keyboard = []
    for i in range(0, len(lang_items), 2):
        row = []
        for code, name in lang_items[i:i+2]:
            row.append(InlineKeyboardButton(name, callback_data=f"dub_{code}"))
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("❌ বাতিল করো", callback_data="dub_cancel")])
    return InlineKeyboardMarkup(keyboard)

def get_start_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📖 কীভাবে ব্যবহার করবো?", callback_data="how_to_use")],
        [
            InlineKeyboardButton("🌐 ভাষার তালিকা", callback_data="lang_list"),
            InlineKeyboardButton("ℹ️ About", callback_data="about")
        ]
    ])

# ══════════════════════════════════════════════
# 🎵  CORE AUDIO PROCESSING
# ══════════════════════════════════════════════
def transcribe_audio_sync(audio_path: str, source_lang: str = None) -> dict:
    """Groq Whisper দিয়ে অডিও transcribe করো (timestamp সহ)"""
    ext = os.path.splitext(audio_path)[1].lower()

    with open(audio_path, 'rb') as f:
        file_content = f.read()

    kwargs = dict(
        file=(f"audio{ext}", file_content),
        model="whisper-large-v3",
        response_format="verbose_json",
        timestamp_granularities=["segment"],
        temperature=0.0
    )
    if source_lang:
        kwargs['language'] = source_lang
    response = groq_mgr.transcribe(**kwargs)

    segments = []
    text = ''

    if hasattr(response, 'segments') and response.segments:
        for s in response.segments:
            if isinstance(s, dict):
                segments.append({
                    'start': float(s.get('start', 0)),
                    'end':   float(s.get('end', 0)),
                    'text':  s.get('text', '').strip()
                })
            else:
                segments.append({
                    'start': float(s.start),
                    'end':   float(s.end),
                    'text':  s.text.strip()
                })
        text = response.text or ' '.join(s['text'] for s in segments)
    elif hasattr(response, 'text'):
        text = response.text

    return {'text': text, 'segments': segments}

def translate_segment_sync(text: str, target_lang: str) -> str:
    """Groq LLaMA দিয়ে টেক্সট অনুবাদ করো"""
    target_name = LANG_FULL_NAMES.get(target_lang, target_lang)

    system_prompt = (
        f"You are a professional dubbing translator. "
        f"Translate the following text to {target_name}. "
        f"Rules:\n"
        f"- Keep the same meaning and natural spoken flow\n"
        f"- Make it sound natural when spoken aloud (dubbing style)\n"
        f"- Do NOT add any explanation, notes, or extra text\n"
        f"- Return ONLY the translated text\n"
        f"- Keep proper nouns (names, places) mostly unchanged\n"
        f"- Match approximate speaking rhythm of the original\n"
        f"- If text is very short (1-3 words), keep translation concise"
    )

    response = groq_mgr.chat(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text}
        ],
        max_tokens=400,
        temperature=0.3
    )

    result = response.choices[0].message.content.strip()
    result = result.strip('"\'`')
    return result

async def generate_tts_segment(text: str, voice: str, output_path: str):
    """Edge TTS দিয়ে টেক্সট থেকে অডিও তৈরি করো"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def adjust_audio_timing(audio: AudioSegment, target_ms: int) -> AudioSegment:
    """অডিওর timing ঠিক করো — speed up / slow down / pad"""
    current_ms = len(audio)

    if current_ms == 0:
        return AudioSegment.silent(duration=max(target_ms, 100))
    if target_ms <= 0:
        return audio

    ratio = current_ms / target_ms

    try:
        if abs(ratio - 1.0) < 0.12:
            if current_ms < target_ms:
                return audio + AudioSegment.silent(duration=target_ms - current_ms)
            return audio[:target_ms]

        elif 0.5 <= ratio <= 2.0:
            adjusted = audio.speedup(playback_speed=ratio, chunk_size=150, crossfade=25)

        elif ratio > 2.0:
            step1 = audio.speedup(playback_speed=2.0, chunk_size=150, crossfade=25)
            remaining = len(step1) / target_ms
            if remaining > 1.1:
                adjusted = step1.speedup(
                    playback_speed=min(remaining, 2.0),
                    chunk_size=150, crossfade=25
                )
            else:
                adjusted = step1
        else:
            adjusted = audio.speedup(playback_speed=max(ratio, 0.5), chunk_size=150, crossfade=25)

        adj_ms = len(adjusted)
        if adj_ms < target_ms:
            return adjusted + AudioSegment.silent(duration=target_ms - adj_ms)
        elif adj_ms > target_ms + 200:
            return adjusted[:target_ms]
        return adjusted

    except Exception as e:
        logger.warning(f"⚠️ Speed adjustment failed: {e} — Fallback")
        if current_ms < target_ms:
            return audio + AudioSegment.silent(duration=target_ms - current_ms)
        return audio[:target_ms]

# ══════════════════════════════════════════════
# 🚀  MAIN DUBBING PIPELINE
# ══════════════════════════════════════════════
async def run_dubbing_pipeline(
    progress_msg,
    bot,
    uid: int,
    audio_info: dict,
    target_lang: str,
    source_lang: str = None
):
    """সম্পূর্ণ ডাবিং pipeline চালাও"""
    lang_name = LANGUAGES[target_lang]
    voice     = TTS_VOICES.get(target_lang, 'en-US-JennyNeural')
    loop      = asyncio.get_event_loop()

    async def update_progress(text: str):
        try:
            await progress_msg.edit_text(text, parse_mode='Markdown')
        except Exception:
            pass

    try:
        # ── STEP 1: Download ──
        await update_progress(
            f"⏳ *ডাবিং শুরু হয়েছে...*\n\n"
            f"🌐 ভাষা: {lang_name}\n"
            f"📁 ফাইল: `{audio_info['file_name']}`\n\n"
            f"🔄 ধাপ ১/৪: অডিও ডাউনলোড হচ্ছে..."
        )

        tg_file = await bot.get_file(audio_info['file_id'])

        ext = os.path.splitext(audio_info['file_name'])[1].lower()
        if not ext or ext not in AUDIO_EXTENSIONS:
            ext = '.ogg'

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, prefix='dub_in_') as tmp:
            await tg_file.download_to_drive(tmp.name)
            audio_path = tmp.name

        logger.info(f"✅ Downloaded: {audio_path}")

        # ── Audio duration ──
        try:
            orig_audio        = AudioSegment.from_file(audio_path)
            total_duration_ms = len(orig_audio)
        except Exception as e:
            logger.error(f"Audio load error: {e}")
            total_duration_ms = (audio_info.get('duration', 60) or 60) * 1000

        dur_sec = total_duration_ms // 1000
        dur_str = f"{dur_sec // 60}:{dur_sec % 60:02d}"

        await update_progress(
            f"⏳ *ডাবিং চলছে...*\n\n"
            f"🌐 ভাষা: {lang_name}\n"
            f"📁 ফাইল: `{audio_info['file_name']}`\n"
            f"⏱️ Duration: {dur_str}\n\n"
            f"✅ ধাপ ১/৪: Download সম্পন্ন!\n"
            f"🔄 ধাপ ২/৪: Speech-to-Text চলছে...\n"
            f"_( Groq Whisper দিয়ে transcribe হচ্ছে )_"
        )

        # ── STEP 2: Transcribe ──
        transcript_data = await loop.run_in_executor(
            executor,
            functools.partial(transcribe_audio_sync, audio_path, source_lang)
        )

        segments  = transcript_data.get('segments', [])
        full_text = transcript_data.get('text', '').strip()

        if not segments and not full_text:
            raise Exception(
                "অডিওতে কোনো কথা খুঁজে পাওয়া যায়নি!\n"
                "পরিষ্কার অডিও দিয়ে আবার চেষ্টা করো।"
            )

        if not segments and full_text:
            segments = [{
                'start': 0.0,
                'end':   total_duration_ms / 1000.0,
                'text':  full_text
            }]

        seg_count = len(segments)
        logger.info(f"✅ Transcribed: {seg_count} segments")

        await update_progress(
            f"⏳ *ডাবিং চলছে...*\n\n"
            f"🌐 ভাষা: {lang_name}\n"
            f"📁 ফাইল: `{audio_info['file_name']}`\n"
            f"⏱️ Duration: {dur_str}\n\n"
            f"✅ ধাপ ১/৪: Download সম্পন্ন!\n"
            f"✅ ধাপ ২/৪: Transcription শেষ! ({seg_count}টি segment)\n"
            f"🔄 ধাপ ৩/৪: অনুবাদ ও TTS তৈরি হচ্ছে...\n"
            f"_( এটি কিছুটা সময় নিতে পারে )_"
        )

        # ── STEP 3: Translate + TTS + Timing ──
        output_audio  = AudioSegment.silent(duration=total_duration_ms)
        success_count = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, seg in enumerate(segments):
                start_ms    = int(seg['start'] * 1000)
                end_ms      = int(seg['end'] * 1000)
                duration_ms = end_ms - start_ms
                seg_text    = seg['text'].strip()

                if not seg_text or duration_ms <= 50:
                    continue

                # Progress bar
                bar_filled = (i + 1) * 10 // seg_count
                bar_empty  = 10 - bar_filled
                pct        = (i + 1) * 100 // seg_count

                if i % 2 == 0 or i == seg_count - 1:
                    try:
                        await progress_msg.edit_text(
                            f"⏳ *ডাবিং চলছে...*\n\n"
                            f"🌐 ভাষা: {lang_name}\n"
                            f"📁 ফাইল: `{audio_info['file_name']}`\n\n"
                            f"✅ Transcription শেষ!\n"
                            f"🔄 Segment: {i + 1}/{seg_count}\n\n"
                            f"`{'█' * bar_filled}{'░' * bar_empty}` {pct}%\n\n"
                            f"💬 _{seg_text[:60]}_",
                            parse_mode='Markdown'
                        )
                    except Exception:
                        pass

                # Translate
                try:
                    translated = await loop.run_in_executor(
                        executor,
                        functools.partial(translate_segment_sync, seg_text, target_lang)
                    )
                    logger.info(f"[{i+1}] '{seg_text[:40]}' → '{translated[:40]}'")
                except Exception as e:
                    logger.error(f"Translation failed for seg {i}: {e}")
                    translated = seg_text

                # TTS generate
                tts_path = os.path.join(tmpdir, f"seg_{i:04d}.mp3")
                try:
                    tts = gTTS(text=translated, lang=voice, slow=False)
                    tts.save(tts_path)
                    if not os.path.exists(tts_path) or os.path.getsize(tts_path) == 0:
                        raise Exception("TTS file empty")
                    tts_audio = AudioSegment.from_file(tts_path)
                    adjusted  = adjust_audio_timing(tts_audio, duration_ms)
                    output_audio  = output_audio.overlay(adjusted, position=start_ms)
                    success_count += 1
                except Exception as e:
                    logger.error(f"TTS failed for seg {i}: {e}")
                    continue

            if success_count == 0:
                raise Exception(f"TTS error — কোনো segment process হয়নি!")

            # ── STEP 4: Export & Send ──
            await update_progress(
                f"⏳ *প্রায় শেষ...*\n\n"
                f"🌐 ভাষা: {lang_name}\n\n"
                f"✅ সব segment তৈরি! ({success_count}/{seg_count})\n"
                f"🔄 ধাপ ৪/৪: ফাইল export ও পাঠানো হচ্ছে..."
            )

            export_path = os.path.join(tmpdir, 'dubbed_final.mp3')
            output_audio.export(export_path, format='mp3', bitrate='128k')

            original_base = os.path.splitext(audio_info['file_name'])[0]
            dubbed_name   = f"{original_base}_dubbed_{target_lang}.mp3"

            final_dur_s = len(output_audio) // 1000

            caption = (
                f"✅ *ডাবিং সম্পন্ন হয়েছে!*\n\n"
                f"🌐 ভাষা: {lang_name}\n"
                f"📊 Segments: {success_count}/{seg_count} ✓\n"
                f"⏱️ Duration: {final_dur_s // 60}:{final_dur_s % 60:02d}\n\n"
                f"_🤖 Groq Whisper + Groq LLaMA + Edge TTS_"
            )

            with open(export_path, 'rb') as af:
                await progress_msg.reply_audio(
                    audio=af,
                    filename=dubbed_name,
                    title=f"Dubbed — {lang_name}",
                    caption=caption,
                    parse_mode='Markdown'
                )

            logger.info(f"✅ Dubbing done for uid {uid} → {lang_name}")

        try:
            await progress_msg.delete()
        except Exception:
            pass

        try:
            os.unlink(audio_path)
        except Exception:
            pass

        user_audio.pop(uid, None)

    except Exception as e:
        logger.error(f"❌ Pipeline error for uid {uid}: {e}")
        try:
            await progress_msg.edit_text(
                f"❌ *ত্রুটি হয়েছে!*\n\n"
                f"`{str(e)[:300]}`\n\n"
                f"আবার চেষ্টা করো অথবা ভিন্ন ফাইল দাও।",
                parse_mode='Markdown'
            )
        except Exception:
            pass
    finally:
        processing.pop(uid, None)

# ══════════════════════════════════════════════
# 🤖  BOT COMMAND HANDLERS
# ══════════════════════════════════════════════
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name
    text = (
        f"🎙️ *Audio Dubbing Bot-এ স্বাগতম, {name}!*\n\n"
        f"আমি তোমার অডিও ফাইল যেকোনো ভাষায় ডাব করে দিতে পারি!\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📤 *ব্যবহার করতে:*\n"
        f"• অডিও ফাইল পাঠাও (MP3 / WAV / OGG / M4A)\n"
        f"• ভাষা বেছে নাও\n"
        f"• ডাব হওয়া অডিও পেয়ে যাও! 🎉\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🌐 *১৪টি ভাষায় ডাবিং:*\n"
        f"বাংলা • English • Hindi • Arabic\n"
        f"French • Spanish • German • Japanese\n"
        f"Korean • Chinese • Russian • Turkish\n"
        f"Italian • Urdu\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ _Groq Whisper + LLaMA + Edge TTS_"
    )
    await update.message.reply_text(
        text, parse_mode='Markdown',
        reply_markup=get_start_keyboard()
    )

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        "📖 *Help — Audio Dubbing Bot*\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📤 *সাপোর্টেড ফরম্যাট:*\n"
        "MP3 • WAV • OGG • M4A • FLAC\n"
        "Voice Message • Opus • AAC\n"
        "📏 সর্বোচ্চ ফাইল সাইজ: ২০MB\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚙️ *কমান্ড:*\n"
        "/start — Bot শুরু করো\n"
        "/help — এই হেল্প দেখো\n"
        "/cancel — বর্তমান কাজ বাতিল করো\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🔄 *কীভাবে কাজ করে:*\n"
        "1️⃣ Groq Whisper দিয়ে Speech-to-Text\n"
        "2️⃣ Groq LLaMA দিয়ে Translation\n"
        "3️⃣ Edge TTS দিয়ে Voice Generation\n"
        "4️⃣ Timing sync করে output তৈরি\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "💡 *টিপস:*\n"
        "• পরিষ্কার অডিওতে ভালো ফলাফল পাবে\n"
        "• Background noise কম থাকলে ভালো\n"
        "• ছোট ফাইলে দ্রুত হয়"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user_audio.pop(uid, None)
    processing.pop(uid, None)
    await update.message.reply_text(
        "❌ *বাতিল করা হয়েছে।*\n\nনতুন অডিও পাঠাতে পারো।",
        parse_mode='Markdown'
    )

# ══════════════════════════════════════════════
# 🎵  AUDIO MESSAGE HANDLER
# ══════════════════════════════════════════════
async def handle_audio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    msg = update.message

    if processing.get(uid):
        await msg.reply_text(
            "⏳ তোমার একটি কাজ এখনো চলছে!\n"
            "অপেক্ষা করো অথবা /cancel দিয়ে বাতিল করো।"
        )
        return

    audio_obj = None
    file_name = 'audio.mp3'
    duration  = 0

    if msg.audio:
        audio_obj = msg.audio
        file_name = msg.audio.file_name or 'audio.mp3'
        duration  = msg.audio.duration or 0
    elif msg.voice:
        audio_obj = msg.voice
        file_name = 'voice_message.ogg'
        duration  = msg.voice.duration or 0
    elif msg.document:
        doc  = msg.document
        mime = doc.mime_type or ''
        name = doc.file_name or ''
        is_audio = (
            mime.startswith('audio/') or
            any(name.lower().endswith(ext) for ext in AUDIO_EXTENSIONS)
        )
        if not is_audio:
            await msg.reply_text(
                "❌ এটা অডিও ফাইল না!\n\n"
                "MP3, WAV, OGG, M4A ইত্যাদি অডিও ফাইল পাঠাও।"
            )
            return
        audio_obj = doc
        file_name = doc.file_name or 'audio.mp3'

    if not audio_obj:
        await msg.reply_text("❌ অডিও ফাইল পাওয়া যায়নি।")
        return

    file_size = getattr(audio_obj, 'file_size', 0) or 0
    if file_size > 20 * 1024 * 1024:
        await msg.reply_text(
            f"❌ ফাইল বড়! ({file_size // (1024*1024)}MB)\n"
            f"সর্বোচ্চ ২০MB এর ফাইল পাঠাও।"
        )
        return

    user_audio[uid] = {
        'file_id':   audio_obj.file_id,
        'file_name': file_name,
        'duration':  duration,
        'size':      file_size
    }

    dur_str  = f"{duration // 60}:{duration % 60:02d}" if duration else "অজানা"
    size_str = f"{file_size // 1024}KB" if file_size < 1024*1024 else f"{file_size//(1024*1024)}MB"

    await msg.reply_text(
        f"🎵 *অডিও পেয়েছি!*\n\n"
        f"📁 ফাইল: `{file_name}`\n"
        f"⏱️ দৈর্ঘ্য: {dur_str}\n"
        f"📏 সাইজ: {size_str}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🗣️ *অডিওতে কোন ভাষায় কথা বলা আছে?*\n"
        f"নিচ থেকে মূল ভাষা সিলেক্ট করো 👇",
        parse_mode='Markdown',
        reply_markup=get_source_language_keyboard()
    )

# ══════════════════════════════════════════════
# 🔘  CALLBACK QUERY HANDLER
# ══════════════════════════════════════════════
async def cb_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    uid  = query.from_user.id
    data = query.data

    if data == "how_to_use":
        text = (
            "📖 *কীভাবে ব্যবহার করবো?*\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "1️⃣ *অডিও ফাইল পাঠাও*\n"
            "   MP3, WAV, OGG, M4A, FLAC\n"
            "   অথবা Voice Message পাঠাও\n\n"
            "2️⃣ *ভাষা সিলেক্ট করো*\n"
            "   Button-এ click করে ভাষা বেছে নাও\n\n"
            "3️⃣ *অপেক্ষা করো*\n"
            "   Bot automatically সব করবে:\n"
            "   • Speech-to-Text (Groq Whisper)\n"
            "   • Translation (Groq LLaMA)\n"
            "   • Voice Generation (Edge TTS)\n"
            "   • Timing Synchronization\n\n"
            "4️⃣ *ডাব হওয়া অডিও পাও!* 🎉\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⏱️ সময়: অডিওর দৈর্ঘ্য অনুযায়ী ১-৫ মিনিট"
        )
        await query.edit_message_text(
            text, parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 ফিরে যাও", callback_data="back_main")
            ]])
        )
        return

    if data == "lang_list":
        lang_text = "🌐 *সাপোর্টেড ভাষার তালিকা:*\n\n"
        for code, name in LANGUAGES.items():
            lang_text += f"• {name}\n"
        lang_text += "\n_সব ভাষায় Timing Sync সহ ডাবিং!_"
        await query.edit_message_text(
            lang_text, parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 ফিরে যাও", callback_data="back_main")
            ]])
        )
        return

    if data == "about":
        text = (
            "ℹ️ *Audio Dubbing Bot*\n\n"
            "🤖 AI দিয়ে অডিও ডাব করার বট\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🛠️ *Technology Stack:*\n"
            "• *Groq Whisper* — Speech Recognition\n"
            "• *Groq LLaMA 3.3 70B* — Translation\n"
            "• *Microsoft Edge TTS* — Voice Synthesis\n"
            "• *PyDub* — Audio Timing Sync\n"
            "• *Python Telegram Bot 20.7*\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🌐 ১৪টি ভাষায় ডাবিং\n"
            "⚡ Fast • 🎯 Accurate • 🔄 Timing Sync"
        )
        await query.edit_message_text(
            text, parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 ফিরে যাও", callback_data="back_main")
            ]])
        )
        return

    if data == "back_main":
        name = query.from_user.first_name
        text = (
            f"🎙️ *Audio Dubbing Bot*\n\n"
            f"অডিও পাঠাও → ভাষা সিলেক্ট করো → ডাব পাও!\n\n"
            f"🌐 ১৪টি ভাষায় Timing Sync সহ ডাবিং"
        )
        await query.edit_message_text(
            text, parse_mode='Markdown',
            reply_markup=get_start_keyboard()
        )
        return

    if data == "dub_cancel":
        user_audio.pop(uid, None)
        await query.edit_message_text(
            "❌ *বাতিল করা হয়েছে।*\n\nনতুন অডিও পাঠাতে পারো।",
            parse_mode='Markdown'
        )
        return

    if data.startswith("src_"):
        src_lang = data[4:]
        audio_info = user_audio.get(uid)
        if not audio_info:
            await query.edit_message_text("❌ অডিও পাওয়া যায়নি। নতুন অডিও পাঠাও।")
            return
        # মূল ভাষা সেভ করো
        audio_info['source_lang'] = src_lang
        src_lang_name = LANGUAGES.get(src_lang, src_lang)
        await query.edit_message_text(
            f"✅ মূল ভাষা: *{src_lang_name}*\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌐 *এখন কোন ভাষায় ডাব করবো?*\n"
            f"নিচ থেকে ভাষা সিলেক্ট করো 👇",
            parse_mode='Markdown',
            reply_markup=get_language_keyboard()
        )
        return

    if data.startswith("dub_"):
        target_lang = data[4:]

        if target_lang not in LANGUAGES:
            await query.answer("❌ অজানা ভাষা!", show_alert=True)
            return

        audio_info = user_audio.get(uid)
        if not audio_info:
            await query.edit_message_text(
                "❌ কোনো অডিও ফাইল নেই!\n\nনতুন অডিও ফাইল পাঠাও।"
            )
            return

        if processing.get(uid):
            await query.answer("⏳ ইতিমধ্যে processing চলছে!", show_alert=True)
            return

        lang_name        = LANGUAGES[target_lang]
        processing[uid]  = True

        progress_msg = await query.edit_message_text(
            f"⏳ *ডাবিং শুরু হচ্ছে...*\n\n"
            f"🌐 ভাষা: {lang_name}\n"
            f"📁 ফাইল: `{audio_info['file_name']}`\n\n"
            f"🔄 প্রস্তুত হচ্ছে...",
            parse_mode='Markdown'
        )

        await run_dubbing_pipeline(
            progress_msg=progress_msg,
            bot=ctx.bot,
            uid=uid,
            audio_info=audio_info,
            target_lang=target_lang,
            source_lang=audio_info.get('source_lang')
        )

# ══════════════════════════════════════════════
# 💬  TEXT HANDLER
# ══════════════════════════════════════════════
async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎵 অডিও ফাইল পাঠাও! আমি ডাব করে দেবো।\n\n"
        "📖 সাহায্যের জন্য: /help\n"
        "🔄 শুরু করতে: /start"
    )

# ══════════════════════════════════════════════
# 🚀  MAIN
# ══════════════════════════════════════════════
def main():
    if not BOT_TOKEN:
        logger.error("❌ BOT_TOKEN missing!")
        return
    if not GROQ_API_KEY:
        logger.error("❌ GROQ_API_KEY missing!")
        return

    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=self_ping, daemon=True).start()
    logger.info("✅ Flask server + Self-ping started")

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .connection_pool_size(16)
        .get_updates_connection_pool_size(8)
        .build()
    )

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CallbackQueryHandler(cb_handler))
    app.add_handler(MessageHandler(
        (filters.AUDIO | filters.VOICE | filters.Document.ALL)
        & filters.ChatType.PRIVATE,
        handle_audio
    ))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE,
        handle_text
    ))

    logger.info("🎙️ Audio Dubbing Bot polling started!")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        read_timeout=30,
        write_timeout=60,
        connect_timeout=30,
        pool_timeout=60
    )

if __name__ == '__main__':
    main()
