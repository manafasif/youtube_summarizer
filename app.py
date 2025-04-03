import os
import re
import sys
import tiktoken
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
import markdown2
from weasyprint import HTML

# --- CONFIG ---
MAX_TOKENS_PER_CHUNK = 2000  # For GPT-4, keep it safe
MODEL = "gpt-4"  # or "gpt-3.5-turbo" for speed/cheaper
# --------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tokenizer = tiktoken.encoding_for_model(MODEL)

def extract_video_id(url):
    if "youtube.com" in url:
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else None
    elif "youtu.be" in url:
        return url.split("/")[-1]
    return url

def fetch_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    lines = []
    for entry in transcript:
        mins, secs = divmod(int(entry['start']), 60)
        timestamp = f"{mins:02}:{secs:02}"
        lines.append(f"[{timestamp}] {entry['text']}")
    return lines

def chunk_transcript(lines, max_tokens):
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

def summarize_chunk(chunk_text, index):
    prompt = f"""
You're a note-taking assistant. Here's part {index+1} of a lecture transcript with timestamps. Summarize the key points in 3–5 bullet points, grouped as a coherent topic section. Include the earliest timestamp.

Transcript:
{chunk_text}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def combine_summaries(summaries):
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
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def save_as_markdown_and_pdf(markdown_text, filename_base):
    md_path = f"{filename_base}.md"
    pdf_path = f"{filename_base}.pdf"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    html = markdown2.markdown(markdown_text)
    HTML(string=html).write_pdf(pdf_path)
    print(f"\n✅ Notes saved as:\n- {md_path}\n- {pdf_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_notes.py <YouTube URL or ID>")
        return

    video_url = sys.argv[1]
    video_id = extract_video_id(video_url)
    print(f"📥 Fetching transcript for: {video_id}")

    transcript_lines = fetch_transcript(video_id)
    chunks = chunk_transcript(transcript_lines, MAX_TOKENS_PER_CHUNK)
    print(f"🔍 Transcript split into {len(chunks)} chunks...")

    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"🧠 Summarizing chunk {i + 1}...")
        summary = summarize_chunk(chunk, i)
        summaries.append(summary)

    print("🧩 Combining summaries...")
    final_notes = combine_summaries(summaries)

    print("💾 Saving notes...")
    save_as_markdown_and_pdf(final_notes, f"summaries/{video_id}_notes")

if __name__ == "__main__":
    main()
