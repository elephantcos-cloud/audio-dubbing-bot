#!/usr/bin/env python3
"""
🎙️ Audio Dubbing Bot — Ultra Version (Edge TTS Fixed)
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

# ── FFmpeg fix for Render ──
import imageio_ffmpeg
from pydub import AudioSegment
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.ffprobe   = imageio_ffmpeg.get_ffmpeg_exe()

# ── Edge TTS Import (gTTS রিমুভ করা হয়েছে) ──
import edge_tts

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

# ── Multiple Groq Keys
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

# ── Edge TTS Neural Voices (সঠিক ইমোশন ও ন্যাচারাল টোনের জন্য) ──
TTS_VOICES = {
    'bn': 'bn-BD-NabanitaNeural', # বাংলার জন্য ন্যাচারাল ভয়েস
    'en': 'en-US-AriaNeural',
    'hi': 'hi-IN-SwaraNeural',
    'ar': 'ar-SA-ZariyahNeural',
    'fr': 'fr-FR-DeniseNeural',
    'es': 'es-ES-ElviraNeural',
    'de': 'de-DE-KatjaNeural',
    'ja': 'ja-JP-NanamiNeural',
    'ko': 'ko-KR-SunHiNeural',
    'zh': 'zh-CN-XiaoxiaoNeural',
    'ru': 'ru-RU-SvetlanaNeural',
    'tr': 'tr-TR-EmelNeural',
    'it': 'it-IT-ElsaNeural',
    'ur': 'ur-PK-UzmaNeural',
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
user_audio = {}   
processing = {}   

# ══════════════════════════════════════════════
# 🌐  FLASK (Keep Alive for Render)
# ══════════════════════════════════════════════
flask_app = Flask(__name__)

@flask_app.route('/')
def home():
    return '🎙️ Audio Dubbing Bot is Running! ✅'

def run_flask():
    port = int(os.environ.get('PORT', 10000))
    flask_app.run(host='0.0.0.0', port=port, debug=False)

def self_ping():
    if not RENDER_URL:
        return
    while True:
        try:
            requests.get(RENDER_URL, timeout=15)
        except Exception:
            pass
        time.sleep(840)

# ══════════════════════════════════════════════
# 🎨  KEYBOARDS
# ══════════════════════════════════════════════
def get_source_language_keyboard():
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
                segments.append({'start': float(s.get('start', 0)), 'end': float(s.get('end', 0)), 'text': s.get('text', '').strip()})
            else:
                segments.append({'start': float(s.start), 'end': float(s.end), 'text': s.text.strip()})
        text = response.text or ' '.join(s['text'] for s in segments)
    elif hasattr(response, 'text'):
        text = response.text

    return {'text': text, 'segments': segments}

def translate_segment_sync(text: str, target_lang: str) -> str:
    target_name = LANG_FULL_NAMES.get(target_lang, target_lang)

    # ── উন্নত প্রম্পট, যাতে অনুবাদ অডিওর সমান থাকে ──
    system_prompt = (
        f"You are a professional dubbing translator. "
        f"Translate the following text to {target_name}. "
        f"Rules:\n"
        f"- EXTREMELY IMPORTANT: Keep the translation VERY concise. "
        f"It must not take longer to speak than the original.\n"
        f"- Make it sound natural when spoken aloud (conversational style).\n"
        f"- Do NOT add any explanation, notes, or quotes.\n"
        f"- Return ONLY the translated text.\n"
        f"- Keep proper nouns (names, places) unchanged."
    )

    response = groq_mgr.chat(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text}
        ],
        max_tokens=400,
        temperature=0.2
    )

    result = response.choices[0].message.content.strip()
    result = result.strip('"\'`')
    return result

async def generate_tts_segment(text: str, voice: str, output_path: str):
    """Edge TTS দিয়ে অডিও জেনারেট"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def adjust_audio_timing(audio: AudioSegment, target_ms: int) -> AudioSegment:
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
            adjusted = step1.speedup(playback_speed=min(remaining, 2.0), chunk_size=150, crossfade=25) if remaining > 1.1 else step1
        else:
            adjusted = audio.speedup(playback_speed=max(ratio, 0.5), chunk_size=150, crossfade=25)

        adj_ms = len(adjusted)
        if adj_ms < target_ms:
            return adjusted + AudioSegment.silent(duration=target_ms - adj_ms)
        elif adj_ms > target_ms + 200:
            return adjusted[:target_ms]
        return adjusted

    except Exception as e:
        logger.warning(f"⚠️ Speed adjustment failed: {e}")
        return audio + AudioSegment.silent(duration=target_ms - current_ms) if current_ms < target_ms else audio[:target_ms]

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
    lang_name = LANGUAGES[target_lang]
    voice     = TTS_VOICES.get(target_lang, 'en-US-AriaNeural')
    loop      = asyncio.get_event_loop()

    async def update_progress(text: str):
        try:
            await progress_msg.edit_text(text, parse_mode='Markdown')
        except Exception:
            pass

    try:
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

        try:
            orig_audio        = AudioSegment.from_file(audio_path)
            total_duration_ms = len(orig_audio)
        except Exception as e:
            logger.error(f"Audio load error: {e}")
            total_duration_ms = (audio_info.get('duration', 60) or 60) * 1000

        dur_sec = total_duration_ms // 1000
        dur_str = f"{dur_sec // 60}:{dur_sec % 60:02d}"

        await update_progress(
            f"✅ ধাপ ১/৪: Download সম্পন্ন!\n"
            f"🔄 ধাপ ২/৪: Speech-to-Text চলছে..."
        )

        transcript_data = await loop.run_in_executor(
            executor,
            functools.partial(transcribe_audio_sync, audio_path, source_lang)
        )

        segments  = transcript_data.get('segments', [])
        full_text = transcript_data.get('text', '').strip()

        if not segments and not full_text:
            raise Exception("অডিওতে কোনো কথা খুঁজে পাওয়া যায়নি!")

        if not segments and full_text:
            segments = [{'start': 0.0, 'end': total_duration_ms / 1000.0, 'text': full_text}]

        seg_count = len(segments)

        await update_progress(
            f"✅ ধাপ ২/৪: Transcription শেষ!\n"
            f"🔄 ধাপ ৩/৪: অনুবাদ ও Edge-TTS তৈরি হচ্ছে...\n"
        )

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

                bar_filled = (i + 1) * 10 // seg_count
                pct        = (i + 1) * 100 // seg_count

                if i % 2 == 0 or i == seg_count - 1:
                    await update_progress(
                        f"🔄 ধাপ ৩/৪: Segment {i + 1}/{seg_count}\n"
                        f"`{'█' * bar_filled}{'░' * (10 - bar_filled)}` {pct}%\n"
                    )

                # Translate
                try:
                    translated = await loop.run_in_executor(
                        executor,
                        functools.partial(translate_segment_sync, seg_text, target_lang)
                    )
                except Exception as e:
                    translated = seg_text

                # ── Edge TTS Integration ──
                tts_path = os.path.join(tmpdir, f"seg_{i:04d}.mp3")
                try:
                    # gTTS এর বদলে Edge-TTS কল করা হচ্ছে
                    await generate_tts_segment(translated, voice, tts_path)
                    
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
                raise Exception("TTS error — কোনো segment process হয়নি!")

            await update_progress(f"🔄 ধাপ ৪/৪: ফাইল export ও পাঠানো হচ্ছে...")

            export_path = os.path.join(tmpdir, 'dubbed_final.mp3')
            output_audio.export(export_path, format='mp3', bitrate='128k')

            original_base = os.path.splitext(audio_info['file_name'])[0]
            dubbed_name   = f"{original_base}_dubbed_{target_lang}.mp3"
            final_dur_s   = len(output_audio) // 1000

            caption = (
                f"✅ *ডাবিং সম্পন্ন হয়েছে!*\n\n"
                f"🌐 ভাষা: {lang_name}\n"
                f"📊 Segments: {success_count}/{seg_count} ✓\n"
                f"⏱️ Duration: {final_dur_s // 60}:{final_dur_s % 60:02d}\n\n"
                f"_🤖 Groq Whisper + Groq LLaMA + Edge TTS_"
            )

            with open(export_path, 'rb') as af:
                await progress_msg.reply_audio(
                    audio=af, filename=dubbed_name, caption=caption, parse_mode='Markdown'
                )

        try:
            await progress_msg.delete()
        except:
            pass
        try:
            os.unlink(audio_path)
        except:
            pass
        user_audio.pop(uid, None)

    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        await update_progress(f"❌ *ত্রুটি হয়েছে!*\n`{str(e)[:300]}`")
    finally:
        processing.pop(uid, None)

# ══════════════════════════════════════════════
# 🤖  BOT HANDLERS
# ══════════════════════════════════════════════
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name
    text = (
        f"🎙️ *Audio Dubbing Bot-এ স্বাগতম, {name}!*\n\n"
        f"আমি তোমার অডিও ফাইল যেকোনো ভাষায় ডাব করে দিতে পারি!\n\n"
        f"📤 *ব্যবহার করতে:*\n"
        f"• অডিও ফাইল পাঠাও (MP3 / WAV / OGG / M4A)\n"
        f"• ভাষা বেছে নাও\n"
        f"• ডাব হওয়া অডিও পেয়ে যাও! 🎉"
    )
    await update.message.reply_text(text, parse_mode='Markdown', reply_markup=get_start_keyboard())

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = "📖 *Help — Audio Dubbing Bot*\n\n/start — Bot শুরু করো\n/cancel — কাজ বাতিল করো\n\nফাইল সাইজ: সর্বোচ্চ ২০MB"
    await update.message.reply_text(text, parse_mode='Markdown')

async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user_audio.pop(uid, None)
    processing.pop(uid, None)
    await update.message.reply_text("❌ *বাতিল করা হয়েছে।*", parse_mode='Markdown')

async def handle_audio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    msg = update.message

    if processing.get(uid):
        await msg.reply_text("⏳ তোমার একটি কাজ এখনো চলছে! অপেক্ষা করো।")
        return

    audio_obj = msg.audio or msg.voice
    file_name = getattr(audio_obj, 'file_name', 'audio.mp3') if audio_obj else 'audio.mp3'
    duration  = getattr(audio_obj, 'duration', 0) if audio_obj else 0

    if msg.document:
        doc = msg.document
        if doc.mime_type.startswith('audio/') or any(doc.file_name.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
            audio_obj, file_name = doc, doc.file_name

    if not audio_obj:
        await msg.reply_text("❌ এটা অডিও ফাইল না!")
        return

    file_size = getattr(audio_obj, 'file_size', 0)
    if file_size > 20 * 1024 * 1024:
        await msg.reply_text("❌ ফাইল বড়! সর্বোচ্চ ২০MB এর ফাইল পাঠাও।")
        return

    user_audio[uid] = {'file_id': audio_obj.file_id, 'file_name': file_name, 'duration': duration, 'size': file_size}

    await msg.reply_text(
        f"🎵 *অডিও পেয়েছি!*\n🗣️ *অডিওতে কোন ভাষায় কথা বলা আছে?*",
        parse_mode='Markdown', reply_markup=get_source_language_keyboard()
    )

async def cb_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid, data = query.from_user.id, query.data

    if data == "how_to_use" or data == "lang_list" or data == "about" or data == "back_main":
        await query.edit_message_text("🔙 মূল মেনুতে ফিরে যাও /start দিন।", parse_mode='Markdown')
        return

    if data == "dub_cancel":
        user_audio.pop(uid, None)
        await query.edit_message_text("❌ *বাতিল করা হয়েছে।*", parse_mode='Markdown')
        return

    if data.startswith("src_"):
        src_lang = data[4:]
        if uid not in user_audio:
            await query.edit_message_text("❌ অডিও পাওয়া যায়নি।")
            return
        user_audio[uid]['source_lang'] = src_lang
        await query.edit_message_text(f"✅ মূল ভাষা সিলেক্টেড।\n🌐 *এখন কোন ভাষায় ডাব করবো?*", parse_mode='Markdown', reply_markup=get_language_keyboard())
        return

    if data.startswith("dub_"):
        target_lang = data[4:]
        if uid not in user_audio:
            await query.edit_message_text("❌ অডিও নেই!")
            return
        if processing.get(uid):
            return

        processing[uid] = True
        progress_msg = await query.edit_message_text("⏳ *প্রস্তুত হচ্ছে...*", parse_mode='Markdown')

        await run_dubbing_pipeline(progress_msg, ctx.bot, uid, user_audio[uid], target_lang, user_audio[uid].get('source_lang'))

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎵 অডিও ফাইল পাঠাও!")

def main():
    if not BOT_TOKEN or not GROQ_API_KEY:
        logger.error("❌ Token/Key missing!")
        return

    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=self_ping, daemon=True).start()

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CallbackQueryHandler(cb_handler))
    app.add_handler(MessageHandler((filters.AUDIO | filters.VOICE | filters.Document.ALL) & filters.ChatType.PRIVATE, handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE, handle_text))

    logger.info("🎙️ Bot polling started!")
    app.run_polling()

if __name__ == '__main__':
    main()
