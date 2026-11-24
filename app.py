"""Lumi.AI: turn a long chapter (PDF, slides, Word, text) into a revision summary,
key sentences and flashcards.

AI tools
  - Gemini (Flash-Lite): writes the summary and flashcards (one request) and
    translates the key sentences (a second, small request)
  - Hugging Face all-MiniLM-L6-v2: picks the most important original sentences
"""
import csv
import html
import io
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import streamlit as st
from pydantic import BaseModel
from PyPDF2 import PdfReader

GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest"]
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS = 120_000  # about 50 pages of text sent to Gemini
MYT = timezone(timedelta(hours=8))  # Malaysia time for the history list

st.set_page_config(page_title="Lumi.AI", page_icon="💡", layout="wide")

# ---------- Styling: notebook paper + ink purple + highlighter yellow ----------
st.markdown("""
<style>
/* Lined notebook paper background with a margin line */
.stApp {
    background-color: #FFFDF6;
    background-image:
        linear-gradient(90deg, transparent 70px, #F6C1CC 70px, #F6C1CC 72px, transparent 72px),
        repeating-linear-gradient(#FFFDF6 0px, #FFFDF6 31px, #E4E9F7 31px, #E4E9F7 32px);
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; max-width: 1300px; }

/* Top bar */
.lumi-topbar {
    display: flex; justify-content: space-between; align-items: center;
    background: #FFFFFF; border: 1px solid #E6E0FA; border-radius: 16px;
    padding: 12px 22px; box-shadow: 0 4px 14px rgba(76, 43, 160, 0.08);
}
.lumi-tagline { color: #5B5675; font-size: 0.95rem; }
.lumi-brand { font-size: 1.7rem; font-weight: 800; color: #4C2BA0; }
.lumi-brand span {
    background: linear-gradient(transparent 55%, #FFE066 55%); padding: 0 4px;
}

/* How-to steps */
.lumi-steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 6px 0 4px; }
.lumi-step {
    background: #FFFFFF; border: 1px solid #E6E0FA; border-radius: 14px; padding: 14px;
}
.lumi-step .num {
    display: inline-block; width: 28px; height: 28px; border-radius: 50%;
    background: #4C2BA0; color: #FFE066; font-weight: 800; text-align: center; line-height: 28px;
}
.lumi-step b { display: block; margin: 8px 0 4px; color: #2B2450; }
.lumi-step p { margin: 0; font-size: 0.88rem; color: #5B5675; }
@media (max-width: 800px) { .lumi-steps { grid-template-columns: repeat(2, 1fr); } }

/* White boxes (Streamlit containers with a key get a .st-key-<key> class) */
.st-key-upload_box, .st-key-result_box, .st-key-settings_box, .st-key-history_box {
    background: #FFFFFF; border-radius: 16px; padding: 18px 20px;
    border: 1px solid #E6E0FA; box-shadow: 0 4px 14px rgba(76, 43, 160, 0.08);
}
.lumi-box-title { font-weight: 800; color: #2B2450; font-size: 1.1rem; margin-bottom: 4px; }
.lumi-hint { color: #6B6685; font-size: 0.88rem; margin-bottom: 8px; }

/* Flashcards */
.lumi-card {
    background: #FAF8FF; border: 1px solid #E6E0FA; border-left: 6px solid #4C2BA0;
    border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;
}
.lumi-card summary { cursor: pointer; font-weight: 700; color: #2B2450; }
.lumi-card .answer {
    margin-top: 10px; padding: 8px 10px; border-radius: 8px;
    background: #FFF4BF; color: #3A3300;
}

/* Key sentences, styled like highlighted text */
.lumi-quote {
    background: #FFFFFF; border-left: 4px solid #FFD23F; border-radius: 10px;
    padding: 10px 14px; margin-bottom: 10px; color: #2B2450;
}
.lumi-quote mark { background: linear-gradient(transparent 60%, #FFE066 60%); color: inherit; }
.lumi-original { margin-top: 6px; font-size: 0.82rem; color: #8A85A3; font-style: italic; }

/* Buttons */
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ---------- Read the file (PDF, PowerPoint, Word, text) ----------
def extract_text(file, max_pages: int):
    """Return (clean text, number of pages/slides read, total, unit name)."""
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        total = len(reader.pages)
        parts = [(p.extract_text() or "") for p in reader.pages[:max_pages]]
        unit = "pages"
    elif name.endswith(".pptx"):
        from pptx import Presentation
        slides = list(Presentation(file).slides)
        total = len(slides)
        parts = []
        for slide in slides[:max_pages]:
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text_frame.text.strip():
                    # slide bullets often have no full stop, so add one
                    parts.append(shape.text_frame.text.strip().rstrip(".") + ".")
        unit = "slides"
    elif name.endswith(".docx"):
        from docx import Document
        parts = [p.text for p in Document(file).paragraphs if p.text.strip()]
        total, unit = 1, "document"
    else:  # .txt
        parts = [file.read().decode("utf-8", errors="ignore")]
        total, unit = 1, "document"
    text = " ".join(" ".join(parts).split())  # also removes line breaks
    return text, min(total, max_pages), total, unit

# ---------- Hugging Face: MiniLM key sentences ----------
@st.cache_resource(show_spinner="Loading Hugging Face model...")
def load_minilm():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MINILM)

def key_sentences(text: str, model, k: int = 8):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if 6 <= len(s.split()) <= 60]
    if len(sentences) <= k:
        return sentences
    vectors = model.encode(sentences, normalize_embeddings=True, batch_size=64)
    centre = vectors.mean(axis=0)            # the chapter's "main idea"
    scores = vectors @ centre                # closeness of each sentence to it
    top = sorted(np.argsort(scores)[-k:])    # best k, kept in original order
    return [sentences[i] for i in top]

# ---------- Gemini: summary + flashcards ----------
DIFFICULTY = {
    "Easy": "simple recall questions about definitions and facts",
    "Medium": "questions that check understanding of how and why things work",
    "Exam-style": "exam-style questions that ask students to explain, compare or apply concepts",
}

def build_prompt(text, language, difficulty, n_cards, quotes):
    return f"""You are a study coach for diploma students.
From the study material below, write in {language}:
1. "summary": a revision summary of at most 350 words that fits on one page, using short paragraphs and bullet points.
2. "flashcards": {n_cards} question-and-answer flashcards. Use {DIFFICULTY[difficulty]}. Each tests one idea. Answers max 2 sentences.
Use only facts from the material.
Return JSON: {{"summary": "...", "flashcards": [{{"question": "...", "answer": "..."}}]}}

MATERIAL:
{text[:MAX_CHARS]}"""

class Flashcard(BaseModel):
    question: str
    answer: str

class StudyPack(BaseModel):  # the exact JSON shape Gemini must return
    summary: str
    flashcards: list[Flashcard]

def ask_gemini(api_key, prompt, schema=None):
    schema = schema or StudyPack
    from google import genai
    from google.genai import errors, types
    client = genai.Client(api_key=api_key)
    for name in GEMINI_MODELS:              # main model, then backups
        for attempt in range(2):            # 2 tries per model
            try:
                resp = client.models.generate_content(
                    model=name,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json",
                                                       response_schema=schema),
                )
                return json.loads(resp.text), name
            except errors.APIError as e:
                if e.code not in (429, 500, 503):   # only retry when busy
                    raise
                time.sleep(2)
    raise RuntimeError("All Gemini models are busy right now. Please try again in a minute.")

class Translation(BaseModel):
    translations: list[str]

def translate_key_sentences(api_key, sentences, language):
    """A separate small Gemini request, so translating is its only job."""
    numbered = chr(10).join(f"{i + 1}. {s}" for i, s in enumerate(sentences))  # chr(10) = new line
    prompt = f"""Translate each of the {len(sentences)} sentences below into {language}.
Return exactly {len(sentences)} items, in the same order. Do not return the English sentence.
Keep acronyms such as NLP unchanged. Only copy a sentence if it is already written in {language}.
Return JSON: {{"translations": ["...", "..."]}}

SENTENCES:
{numbered}"""
    data, _ = ask_gemini(api_key, prompt, Translation)
    return data["translations"]

def cards_to_csv(cards):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Question", "Answer"])  # imports into Anki / Quizlet
    for c in cards:
        writer.writerow([c["question"], c["answer"]])
    return buf.getvalue()

# ---------- Session state: history of study packs ----------
st.session_state.setdefault("history", [])     # list of study packs
st.session_state.setdefault("current", None)   # index of the pack being viewed
st.session_state.setdefault("uploader_key", 0)  # change it to clear the upload box

def view_pack(i):
    st.session_state.current = i

def new_notes():
    st.session_state.current = None
    st.session_state.uploader_key += 1

def clear_history():
    st.session_state.history = []
    new_notes()

def render_pack(pack, idx):
    st.markdown(f'<div class="lumi-box-title">📘 {html.escape(pack["name"])}</div>'
                f'<div class="lumi-hint">{html.escape(pack["info"])}</div>',
                unsafe_allow_html=True)
    tab_sum, tab_quotes, tab_cards = st.tabs(
        ["📝 Summary", "📌 Key sentences", "🃏 Flashcards"])

    with tab_sum:
        st.markdown('<div class="lumi-hint">A one-page overview of the whole chapter, '
                    'written by Gemini. Read this first.</div>', unsafe_allow_html=True)
        st.markdown(pack["summary"])
        st.download_button("⬇️ Download summary", pack["summary"],
                           "lumi_summary.txt", key=f"dl_sum_{idx}")

    with tab_quotes:
        st.markdown('<div class="lumi-hint">The most important sentences from your file, '
                    'picked by a Hugging Face model and translated into your chosen '
                    'language. The original wording is shown underneath, so you can '
                    'double-check the AI.</div>', unsafe_allow_html=True)
        if pack.get("note"):
            st.warning(pack["note"])
        for q in pack["quotes"]:
            original = ""
            if q["translated"].strip() != q["original"].strip():
                original = (f'<div class="lumi-original">Original: '
                            f'{html.escape(q["original"])}</div>')
            st.markdown(f'<div class="lumi-quote"><mark>{html.escape(q["translated"])}</mark>'
                        f'{original}</div>', unsafe_allow_html=True)

    with tab_cards:
        st.markdown('<div class="lumi-hint">Try to answer each question in your head, '
                    'then click it to check the answer.</div>', unsafe_allow_html=True)
        for i, c in enumerate(pack["cards"], 1):
            st.markdown(
                f'<div class="lumi-card"><details>'
                f'<summary>Q{i}. {html.escape(c["question"])}</summary>'
                f'<div class="answer">{html.escape(c["answer"])}</div>'
                f'</details></div>',
                unsafe_allow_html=True)
        st.download_button("⬇️ Download flashcards (CSV for Anki/Quizlet)",
                           cards_to_csv(pack["cards"]), "lumi_flashcards.csv",
                           "text/csv", key=f"dl_cards_{idx}")

# ---------- Top bar: tagline on the left, Lumi.AI on the right ----------
st.markdown("""
<div class="lumi-topbar">
  <div class="lumi-tagline">▪ Your AI revision buddy: long notes in, quick revision out.</div>
  <div class="lumi-brand">💡 <span>Lumi.AI</span></div>
</div>
""", unsafe_allow_html=True)

# ---------- How to use (open until the first study pack is made) ----------
with st.expander("👋 New here? How to use Lumi.AI", expanded=not st.session_state.history):
    st.markdown("""
<div class="lumi-steps">
  <div class="lumi-step"><span class="num">1</span><b>Upload your notes</b>
    <p>Drop in a chapter as a PDF, PowerPoint (.pptx), Word (.docx) or text file.</p></div>
  <div class="lumi-step"><span class="num">2</span><b>Pick your settings</b>
    <p>On the right, choose English or Bahasa Melayu, flashcard difficulty and how many cards.</p></div>
  <div class="lumi-step"><span class="num">3</span><b>Click Generate</b>
    <p>Lumi reads the whole file in seconds. Long chapters may take up to a minute.</p></div>
  <div class="lumi-step"><span class="num">4</span><b>Revise</b>
    <p>Read the summary, check the key sentences, then test yourself with the flashcards.</p></div>
</div>
""", unsafe_allow_html=True)
    st.markdown("**What you get:** 📝 a one-page **Summary** · 📌 **Key sentences** quoted "
                "from your file · 🃏 **Flashcards** to test yourself. Every study pack is "
                "saved in **History** on the right, so you can upload new notes and still "
                "go back to old ones.")

left, right = st.columns([2.3, 1], gap="large")

# ---------- Right column: Settings (always visible) + History ----------
with right:
    with st.container(key="settings_box"):
        st.markdown('<div class="lumi-box-title">⚙️ Settings</div>'
                    '<div class="lumi-hint">Choose these before you click Generate.</div>',
                    unsafe_allow_html=True)
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            api_key = st.text_input("Gemini API key", type="password",
                                    help="Free key at https://aistudio.google.com/apikey")
        language = st.radio("Language", ["English", "Bahasa Melayu"], horizontal=True)
        difficulty = st.select_slider("Flashcard difficulty", list(DIFFICULTY), value="Medium",
                                      help="Easy = facts, Medium = understanding, "
                                           "Exam-style = explain and apply")
        n_cards = st.slider("Number of flashcards", 5, 25, 10)
        max_pages = st.number_input("Max pages / slides to read", 1, 300, 60)

    st.write("")
    with st.container(key="history_box"):
        st.markdown('<div class="lumi-box-title">🕘 History</div>', unsafe_allow_html=True)
        history = st.session_state.history
        if not history:
            st.markdown('<div class="lumi-hint">Your study packs will appear here.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="lumi-hint">Click one to open it again. History is '
                        'kept until you close or refresh this page.</div>',
                        unsafe_allow_html=True)
            for i in reversed(range(len(history))):
                p = history[i]
                active = i == st.session_state.current
                st.button(f"{'▶ ' if active else ''}{p['name']}  ·  {p['time']}",
                          key=f"hist_{i}", on_click=view_pack, args=(i,),
                          type="primary" if active else "secondary",
                          width="stretch")
            st.button("🗑️ Clear history", on_click=clear_history, key="clear_hist")

# ---------- Left column: Upload + Results ----------
with left:
    with st.container(key="upload_box"):
        top_l, top_r = st.columns([3, 1])
        top_l.markdown('<div class="lumi-box-title">📄 Upload your notes</div>'
                       '<div class="lumi-hint">PDF, PowerPoint (.pptx), Word (.docx) '
                       'or text (.txt)</div>', unsafe_allow_html=True)
        top_r.button("➕ New notes", on_click=new_notes, key="new_notes",
                     width="stretch")
        uploaded = st.file_uploader("Upload", type=["pdf", "pptx", "docx", "txt"],
                                    key=f"up_{st.session_state.uploader_key}",
                                    label_visibility="collapsed")
        generate = st.button("· Generate study pack", type="primary",
                             disabled=uploaded is None, width="stretch")

    if generate:
        if not api_key:
            st.error("Please enter a Gemini API key in Settings.")
            st.stop()
        start = time.time()
        with st.spinner("📖 Reading your file..."):
            text, used, total, unit = extract_text(uploaded, max_pages)
        if len(text.split()) < 100:
            st.error("Almost no text found. The file may be scanned images, or too short.")
            st.stop()
        try:
            with st.status("Lumi is studying your notes...", expanded=True) as status:
                st.write(f"📖 Read {used} {unit} ({len(text.split()):,} words)")
                st.write("📌 Finding key sentences (Hugging Face MiniLM)...")
                originals = key_sentences(text, load_minilm())
                st.write("🧠 Writing summary and flashcards (Gemini)...")
                result, model_used = ask_gemini(
                    api_key, build_prompt(text, language, difficulty, n_cards, originals))
                st.write("🌏 Translating key sentences (Gemini)...")
                try:
                    translated = [str(t) for t in
                                  translate_key_sentences(api_key, originals, language)]
                except Exception:   # keep the pack even if translating fails
                    translated = []
                note = ""
                if len(translated) != len(originals):
                    note = (f"Gemini translated {len(translated)} of {len(originals)} key "
                            "sentences. Any missing ones are shown in the original language.")
                # use each translation if there is one, otherwise keep the original
                quotes = [{"original": o,
                           "translated": translated[i] if i < len(translated) else o}
                          for i, o in enumerate(originals)]
                status.update(label=f"Done in {time.time() - start:.1f} seconds!",
                              state="complete", expanded=False)
            st.session_state.history.append({
                "name": uploaded.name,
                "time": datetime.now(MYT).strftime("%d %b, %I:%M %p"),
                "summary": result["summary"],
                "cards": result["flashcards"],
                "quotes": quotes,
                "note": note,
                "info": f"{used} of {total} {unit} · {len(text.split()):,} words · "
                        f"{time.time() - start:.1f} s · {language} · {difficulty} · "
                        f"Gemini: {model_used}",
            })
            st.session_state.current = len(st.session_state.history) - 1
            done = True
        except Exception as e:  # show API errors on the page
            st.error(f"Something went wrong: {e}")
            done = False
        if done:
            st.rerun()  # refresh so the new pack shows in History

    st.write("")
    current = st.session_state.current
    with st.container(key="result_box"):
        if current is None or current >= len(st.session_state.history):
            st.markdown('<div class="lumi-box-title">· Your study pack</div>'
                        '<div class="lumi-hint">Upload a file and click <b>Generate study '
                        'pack</b>. Your summary, key sentences and flashcards will appear '
                        'here.</div>', unsafe_allow_html=True)
        else:
            render_pack(st.session_state.history[current], current)