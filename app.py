import os
import re
import streamlit as st
import tiktoken
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
import markdown2
from weasyprint import HTML

# --- CONFIG ---
DEFAULT_MAX_TOKENS = 6000  # For GPT-4, keep it safe
DEFAULT_MODEL = "gpt-3.5-turbo-16k"  # or "gpt-3.5-turbo" for speed/cheaper
# --------------

# Initialize OpenAI client
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY environment variable not set.")
        return None
    return OpenAI(api_key=api_key)

# Utility functions
def extract_video_id(url: str) -> str:
    if "youtube.com" in url:
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else None
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return url

@st.cache_data
def fetch_transcript(video_id: str) -> list[str]:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    lines = []
    for entry in transcript:
        mins, secs = divmod(int(entry['start']), 60)
        timestamp = f"{mins:02}:{secs:02}"
        lines.append(f"[{timestamp}] {entry['text']}")
    return lines

@st.cache_data
def chunk_transcript(lines: list[str], max_tokens: int, model: str) -> list[str]:
    tokenizer = tiktoken.encoding_for_model(model)
    chunks, current_chunk = [], []
    token_count = 0
    for line in lines:
        line_tokens = len(tokenizer.encode(line))
        if token_count + line_tokens > max_tokens:
            chunks.append("\n".join(current_chunk))
            current_chunk, token_count = [], 0
        current_chunk.append(line)
        token_count += line_tokens
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

@st.cache_data
def summarize_chunk(_client: OpenAI, chunk_text: str, index: int, model: str) -> str:
    client = _client
    prompt = f"""
You're a note-taking assistant. Here's part {index+1} of a lecture transcript with timestamps. Summarize the key points into detailed bullet points, grouped as a coherent topic section. Include the earliest timestamp.

Transcript:
{chunk_text}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

@st.cache_data
def combine_summaries(_client: OpenAI, summaries: list[str], model: str) -> str:
    client = _client
    prompt = f"""
You are an AI assistant combining summarized lecture notes from different sections of a talk.

Here are the notes from each chunk:
{chr(10).join(summaries)}

Format the final output as:
- Markdown
- Timestamps at the start of each section
- Topic headers
- Clear bullet points

Begin:
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# Streamlit App UI
st.title("YouTube Lecture Notes Generator")  # Do NOT reassign st

# Sidebar settings
model = st.sidebar.selectbox("OpenAI Model", ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"], index=1)
max_tokens = st.sidebar.number_input("Max tokens per chunk", min_value=1000, max_value=12000, value=DEFAULT_MAX_TOKENS, step=500)

video_input = st.text_input("YouTube URL or Video ID")
if st.button("Generate Notes"):
    if not video_input:
        st.warning("Please enter a YouTube URL or ID.")
    else:
        client = get_openai_client()
        if not client:
            st.stop()
        video_id = extract_video_id(video_input)
        if not video_id:
            st.error("Could not extract video ID. Please check the URL or ID.")
            st.stop()
        st.info(f"Fetching transcript for video: {video_id}")
        with st.spinner("Fetching transcript..."):
            try:
                transcript_lines = fetch_transcript(video_id)
            except Exception as e:
                st.error(f"Error fetching transcript: {e}")
                st.stop()
        st.success(f"Transcript fetched: {len(transcript_lines)} lines.")

        with st.spinner("Chunking transcript..."):
            chunks = chunk_transcript(transcript_lines, max_tokens, model)
        st.success(f"Transcript split into {len(chunks)} chunks.")

        summaries = []
        progress = st.progress(0)
        for i, chunk in enumerate(chunks):
            with st.spinner(f"Summarizing chunk {i+1} of {len(chunks)}..."):
                summary = summarize_chunk(client, chunk, i, model)
            summaries.append(summary)
            progress.progress((i + 1) / len(chunks))
        st.success("All chunks summarized.")

        with st.spinner("Combining summaries..."):
            final_notes = combine_summaries(client, summaries, model)
        st.success("Final notes generated.")

        # Display and Download
        st.markdown("---")
        st.header("📖 Final Lecture Notes")
        st.markdown(final_notes)

        # Prepare downloads
        md_bytes = final_notes.encode("utf-8")
        html = markdown2.markdown(final_notes)
        pdf_bytes = HTML(string=html).write_pdf()

        st.download_button(
            "📥 Download as Markdown",
            md_bytes,
            file_name=f"{video_id}_notes.md",
            mime="text/markdown"
        )
        st.download_button(
            "📥 Download as PDF",
            pdf_bytes,
            file_name=f"{video_id}_notes.pdf",
            mime="application/pdf"
        )
