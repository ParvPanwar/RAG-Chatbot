# Social Media Video Comparison RAG Chatbot

I built this as a working prototype for comparing short-form/video content with a RAG chatbot on top of the transcript.

The idea is simple: give the app two videos from the same platform, let it pull the transcript and metadata, then ask questions like why one performed better, how the hooks compare, or what should be changed in the weaker video.

I originally started with cross-platform comparison, but while testing it became clear that YouTube vs Instagram is not a clean comparison. The metrics behave differently and the available metadata is different. So I changed the app to compare:

- YouTube video vs YouTube video
- Instagram Reel vs Instagram Reel

That made the comparison more useful and easier to explain.

## What Works Right Now

- Analyze two YouTube videos
- Analyze two Instagram Reels
- Extract YouTube transcripts with `youtube-transcript-api`
- Fall back to Whisper transcription if YouTube captions are not available
- Download Instagram audio with `yt-dlp`
- Transcribe Instagram audio with `faster-whisper`
- Pull metadata like title, creator, likes, comments, duration, upload date, hashtags, and views when available
- Use Apify for Instagram metadata because `yt-dlp` was not reliably returning Reel views
- Calculate engagement rate when views are available
- Chunk transcripts and store them in ChromaDB
- Ask questions through a RAG chatbot
- Stream chat responses
- Show citations by video and chunk
- Keep conversation memory for follow-up questions

## Demo Flow I Use

1. Start the backend.
2. Start the frontend.
3. Pick YouTube or Instagram.
4. Paste two links from the same platform.
5. Click Analyze.
6. Check the two video cards.
7. Ask the chatbot:
   - Why did Video A perform better?
   - Compare the hooks in the first 5 seconds.
   - What is the engagement rate of each video?
   - Suggest improvements for Video B based on Video A.

## Tech Stack

Frontend:

- Next.js
- React
- TypeScript

Backend:

- FastAPI
- Pydantic
- yt-dlp
- youtube-transcript-api
- faster-whisper

RAG:

- LangChain
- ChromaDB
- local HuggingFace embeddings
- remote Ollama-compatible chat model

External metadata:

- Apify for Instagram metadata
- yt-dlp as fallback

## How The App Works

When the user submits two URLs, the backend creates one analysis session.

For YouTube:

1. Extract the YouTube video ID.
2. Pull title, creator, views, likes, comments, upload date, duration, hashtags, and subscriber count when available.
3. Fetch captions using `youtube-transcript-api`.
4. If captions fail, download audio and transcribe it with Whisper.

For Instagram:

1. Extract the Reel ID.
2. Try Apify first for metadata.
3. Fall back to `yt-dlp` metadata if Apify is not configured or fails.
4. Download audio with `yt-dlp`.
5. Transcribe audio with `faster-whisper`.
6. Delete the temporary audio after processing.

After both videos are processed:

1. Calculate engagement rate:

```text
(likes + comments) / views * 100
```

2. Split transcripts into chunks.
3. Store chunks in ChromaDB with metadata like:

```text
analysis_id
video_id
video_label
chunk_id
title
creator
platform
start_time
end_time
```

4. Use retrieved chunks as context for the chatbot.
5. Stream the answer back to the frontend.

## Why I Added Apify

This was one of the bigger fixes during testing.

At first, Instagram metadata only used `yt-dlp`. It worked for titles, likes, comments, duration, and sometimes creator names, but Reel views were often missing.

That was a problem because views are required for engagement rate. If views are missing, the app cannot honestly calculate:

```text
(likes + comments) / views * 100
```

So I added Apify as the primary Instagram metadata source.

The current Instagram metadata flow is:

1. Try Apify.
2. If Apify returns metadata, use it.
3. If Apify fails, use `yt-dlp` fallback.
4. Keep `yt-dlp` for audio download because it already worked well for that.

This is not perfect, but it made Instagram testing much more reliable.

## Problems I Hit While Building

These are the main issues I had to fix:

### Duplicate Chroma IDs

At one point ChromaDB failed because chunks were being inserted with duplicate IDs.

Fix:

- include the analysis ID, platform, video ID, and chunk number in the stored chunk ID
- keep chunks unique per analysis run

### Instagram Views Missing

`yt-dlp` did not reliably return Reel views.

Fix:

- add Apify for Instagram metadata
- keep `yt-dlp` as fallback

### Chat Answered Random Questions

The chatbot was answering even when the user typed random text that was not really about the videos.

Fix:

- tighten the chat prompt
- make it ask for a clear video-related question when the input is too vague
- keep answers grounded in transcript or metadata

### Markdown Tables Broke The Chat UI

Some answers came back as markdown tables, but the table overflow made the chat panel awkward to scroll.

Fix:

- add markdown table styling
- allow only the table area to scroll horizontally
- stop the whole chat bubble from clipping normal text

### Citation Dropdown Scroll

Opening a citation sometimes made the transcript text hard to scroll.

Fix:

- make open citation sections scroll independently
- keep the chat panel stable

### Instagram Reliability

Some Reels work cleanly, some do not. This mostly depends on whether Instagram/Apify/yt-dlp can access the public metadata and media.

Fix:

- add clearer error handling
- keep fallback paths
- show unavailable fields instead of guessing

## RAG Behavior

The chatbot is designed to answer from:

- metadata
- retrieved transcript chunks
- comparison metrics
- conversation memory

It should cite transcript-based answers like this:

```text
[Video A - Chunk A-3]
```

For metadata answers, it can answer without a transcript citation because the value comes from the video metadata card.

For creative suggestions, the app can give grounded recommendations, but it should not pretend to predict the future perfectly.

## Current Limitations

- Instagram views depend on Apify returning that field.
- Follower/subscriber counts are not always available.
- Whisper can make mistakes if there is music, noise, or overlapping speech.
- The backend stores analysis results and chat memory in process memory, so restarting the server clears them.
- The app is built for local testing, not production traffic.
- Forecasting performance is not reliable without historical creator data.
- ChromaDB is local right now.

## Cost And Scale

The current version is cost-optimized for local testing by using local embeddings, local transcription, ChromaDB, and a remote Ollama-compatible model.

It is not yet production-ready for 1,000+ creators daily. To reach that scale, I would add background workers, caching, GPU transcription, persistent storage, job queues, and cost monitoring.

## Run Locally

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

Create `.env` from `.env.example` and add your keys.

Important values:

```env
OLLAMA_BASE_URL=
OLLAMA_CHAT_MODEL=
OLLAMA_API_KEY=

APIFY_API_TOKEN=
APIFY_INSTAGRAM_ACTOR_ID=apify/instagram-scraper
APIFY_TIMEOUT_SECONDS=90

LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

Start backend:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Start frontend:

```bash
cd frontend
npm run dev
```

Open:

```text
http://localhost:3000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Project Structure

```text
app/
  routes/
  schemas/
  services/
    youtube_analyzer.py
    instagram_analyzer.py
    apify_instagram.py
    combined_analysis.py
    vector_store.py
    retriever.py
    rag_chat.py

frontend/
  app/
    page.tsx
    globals.css

requirements.txt
.env.example
```

## What I Would Improve Next

If I continued this project, I would add:

- persistent database storage for analyses
- background jobs for long video processing
- better creator history tracking
- a ranking/scoring step during retrieval so the best transcript chunks are prioritized before sending context to the LLM
- smarter chunking that separates hooks, CTAs, topic shifts, and high-energy moments instead of only splitting by text length
- stronger prediction support using past videos from the same creator
- a structured comparison report before the chat starts
- deployment setup

## How I Would Make It Production Ready

Right now this is a local prototype. If I wanted to turn it into something that could handle real creator usage, I would not just deploy the current backend as-is. I would change the architecture in phases.

### 1. Move video analysis into background jobs

The current `/analyze` request does a lot of work directly: metadata extraction, audio download, transcription, chunking, embedding, and indexing.

For production, I would move that into a worker system with something like Redis plus Celery or RQ.

The API would return a job ID first, then the frontend would show progress while the worker processes the videos.

### 2. Add persistent storage

Right now analysis results and chat memory are stored in process memory. That is okay for local testing, but it disappears when the backend restarts.

For production, I would add PostgreSQL to store:

- users
- analysis sessions
- video metadata
- transcript status
- chat sessions
- usage history

### 3. Use a scalable vector database

Local ChromaDB is fine for development, but I would not use it as the main production vector database for many users.

I would move to hosted Chroma, Qdrant, Pinecone, Weaviate, or pgvector depending on cost and deployment needs.

### 4. Cache expensive work

The same video should not be downloaded, transcribed, embedded, and indexed again and again.

I would cache:

- metadata by video URL/video ID
- transcripts
- embeddings
- completed analysis results

This would help a lot with both speed and cost.

### 5. Improve transcription performance

Whisper on CPU is useful for testing, but it can be slow.

For higher traffic, I would use GPU workers or a managed transcription service, depending on cost. I would also put limits on max video duration so one user cannot block the system with a very long video.

### 6. Add auth and usage limits

For a real product, I would add login and per-user limits.

This would make it possible to track:

- how many videos a user analyzed
- how much LLM usage they consumed
- failed jobs
- retry attempts
- billing or free-tier limits

### 7. Add monitoring and cost tracking

For 1,000+ creators daily, I would need to know where time and money are going.

I would track:

- analysis time
- transcription time
- Apify usage
- LLM token usage
- failed Instagram jobs
- vector database latency
- backend errors

Without this, the app might work technically but the cost could still get out of control.

### 8. Harden deployment

Before calling it production-ready, I would Dockerize the backend, separate staging and production env files, restrict CORS properly, add CI checks, and deploy frontend and backend separately.

The likely setup would be:

```text
Frontend: Vercel or Netlify
Backend: Render, Railway, Fly.io, or a VPS
Database: PostgreSQL
Queue: Redis
Workers: CPU/GPU worker machines
Vector DB: hosted Chroma, Qdrant, Pinecone, Weaviate, or pgvector
```

So the production version is possible, but it needs job queues, caching, persistent storage, scaling, and monitoring. The current version is best described as a working prototype that proves the comparison and RAG flow.

## Final Status

The project is ready for local testing with real YouTube videos and public Instagram Reels.

The main thing I would mention during a demo is that the RAG chatbot is not just answering from a prompt. It is using transcript chunks, metadata, comparison metrics, citations, and memory together.
