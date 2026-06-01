"use client";

import { FormEvent, PointerEvent, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowUpRight,
  Clapperboard,
  Loader2,
  MessageSquare,
  Play,
  Send,
  Sparkles,
  VolumeX,
} from "lucide-react";

type Metadata = {
  title?: string | null;
  creator?: string | null;
  follower_count?: number | null;
  hashtags?: string[];
  views?: number | null;
  likes?: number | null;
  comments?: number | null;
  upload_date?: string | null;
  duration?: number | null;
};

type VideoAnalysis = {
  platform: string;
  video_id: string;
  video_label?: string | null;
  url: string;
  metadata: Metadata;
  transcript: { text: string; start: number; duration: number }[];
  engagement_rate?: number | null;
};

type ComparisonSummary = {
  higher_engagement_platform?: string | null;
  engagement_rate_gap?: number | null;
  duration_difference_seconds?: number | null;
};

type AnalyzeResponse = {
  analysis_id: string;
  video_a: VideoAnalysis;
  video_b: VideoAnalysis;
  comparison?: ComparisonSummary | null;
  chunks_indexed?: number | null;
};

type Citation = {
  citation: string;
  video_id: string;
  title?: string | null;
  creator?: string | null;
  chunk_id: string;
  chunk_text: string;
  start_time?: number | null;
  end_time?: number | null;
};

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  citations?: Citation[];
};

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ??
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  "http://127.0.0.1:8000"; // local FastAPI fallback

const MIN_CHAT_WIDTH = 340;
const MAX_CHAT_WIDTH = 760;

export default function Home() {
  const [platform, setPlatform] = useState<"youtube" | "instagram">("youtube");
  const [urlA, setUrlA] = useState("");
  const [urlB, setUrlB] = useState("");
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatWidth, setChatWidth] = useState(410);
  const activeRequest = useRef<AbortController | null>(null);

  const chatReady = Boolean(analysis?.analysis_id) && !streaming; // block overlap while one answer streams

  async function handleAnalyze(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setAnalyzing(true);
    setMessages([]);
    setConversationId(null);
    setAnalysis(null);

    try {
      const response = await fetch(`${BACKEND_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          platform,
          video_url_a: urlA,
          video_url_b: urlB,
        }),
      });

      if (!response.ok) {
        throw new Error(await readApiError(response));
      }

      const data = (await response.json()) as AnalyzeResponse;
      setAnalysis(data);
    } catch (err) {
      if (err instanceof TypeError && err.message.toLowerCase().includes("failed to fetch")) {
        setError(
          "Cannot reach the backend server. Make sure FastAPI is running: source .venv/bin/activate && uvicorn app.main:app --reload"
        );
      } else {
        setError(err instanceof Error ? err.message : "Analysis failed.");
      }
    } finally {
      setAnalyzing(false);
    }
  }

  async function handleChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!analysis || !chatInput.trim()) return;

    const question = chatInput.trim();
    setChatInput("");
    setError(null);
    setStreaming(true);
    setMessages((current) => [
      ...current,
      { role: "user", text: question },
      { role: "assistant", text: "", citations: [] },
    ]);

    const controller = new AbortController();
    activeRequest.current = controller;

    try {
      const response = await fetch(`${BACKEND_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          analysis_id: analysis.analysis_id,
          message: question,
          conversation_id: conversationId,
        }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(await readApiError(response));
      }

      await readSseStream(response.body, {
        onConversation: setConversationId,
        onToken: (token) => {
          setMessages((current) => updateLastAssistant(current, token));
        },
        onCitations: (citations) => {
          setMessages((current) => attachCitations(current, citations));
        },
        onError: (message) => {
          setError(message);
          setMessages((current) => replaceLastAssistant(current, message));
        },
      });
    } catch (err) {
      if (err instanceof TypeError && err.message.toLowerCase().includes("failed to fetch")) {
        setError(
          "Cannot reach the backend server. Make sure FastAPI is running: source .venv/bin/activate && uvicorn app.main:app --reload"
        );
      } else if (!(err instanceof DOMException && err.name === "AbortError")) {
        setError(err instanceof Error ? err.message : "Chat failed.");
      }
    } finally {
      setStreaming(false);
      activeRequest.current = null;
    }
  }

  const statusLabel = useMemo(() => {
    if (analyzing) return "Analyzing both videos...";
    if (streaming) return "Streaming answer...";
    if (analysis) return "Analysis ready";
    return "Waiting for two URLs";
  }, [analysis, analyzing, streaming]);

  function handleChatResizeStart(event: PointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    const pointerId = event.pointerId;
    const handle = event.currentTarget;
    handle.setPointerCapture(pointerId);
    document.body.classList.add("isResizingChat");

    function updateWidth(clientX: number) {
      // Right edge stays fixed; dragging changes only the chat column.
      const nextWidth = window.innerWidth - clientX;
      setChatWidth(clamp(nextWidth, MIN_CHAT_WIDTH, Math.min(MAX_CHAT_WIDTH, window.innerWidth - 560)));
    }

    function handlePointerMove(moveEvent: globalThis.PointerEvent) {
      updateWidth(moveEvent.clientX);
    }

    function handlePointerUp() {
      document.body.classList.remove("isResizingChat");
      handle.releasePointerCapture(pointerId);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
    }

    updateWidth(event.clientX);
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("pointercancel", handlePointerUp);
  }

  return (
    <main
      className="shell"
      style={{ "--chat-width": `${chatWidth}px` } as React.CSSProperties}
    >
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="kicker">Video Compare Desk</p>
            <h1>Compare social media videos, with the receipts.</h1>
          </div>
          <div className="status">
            <span className={`statusDot ${analyzing || streaming ? "pulse" : ""}`} />
            {statusLabel}
          </div>
        </header>

        <form className="urlForm" onSubmit={handleAnalyze}>
          <label className="platformSelectorLabel">
            <span>Platform</span>
            <select
              value={platform}
              onChange={(event) => {
                setPlatform(event.target.value as "youtube" | "instagram");
                setUrlA("");
                setUrlB("");
              }}
              disabled={analyzing}
              className="platformSelect"
            >
              <option value="youtube">YouTube</option>
              <option value="instagram">Instagram</option>
            </select>
          </label>
          <label>
            <span>{platform === "youtube" ? "YouTube URL A" : "Instagram Reel URL A"}</span>
            <input
              value={urlA}
              onChange={(event) => setUrlA(event.target.value)}
              placeholder={platform === "youtube" ? "https://www.youtube.com/watch?v=..." : "https://www.instagram.com/reel/..."}
              disabled={analyzing}
              required
            />
          </label>
          <label>
            <span>{platform === "youtube" ? "YouTube URL B" : "Instagram Reel URL B"}</span>
            <input
              value={urlB}
              onChange={(event) => setUrlB(event.target.value)}
              placeholder={platform === "youtube" ? "https://www.youtube.com/watch?v=..." : "https://www.instagram.com/reel/..."}
              disabled={analyzing}
              required
            />
          </label>
          <button className="analyzeButton" disabled={analyzing}>
            {analyzing ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            Analyze
          </button>
        </form>

        {error ? (
          <div className="error animated-slide-in">
            <span className="errorText">{error}</span>
            <button className="errorClose" onClick={() => setError(null)} aria-label="Close error">
              ×
            </button>
          </div>
        ) : null}

        {analysis?.comparison && (
          <article className="comparisonSummaryCard animated-fade-in">
            <div className="summaryBadge">
              <Sparkles className="badgeIcon" size={16} />
              <span>
                {analysis.comparison.higher_engagement_platform === "A"
                  ? "Video A Leads Engagement Rate!"
                  : analysis.comparison.higher_engagement_platform === "B"
                  ? "Video B Leads Engagement Rate!"
                  : analysis.comparison.higher_engagement_platform === "tie"
                  ? "Engagement Rate Tie!"
                  : "Engagement Comparison Complete"}
              </span>
            </div>
            <div className="summaryMetrics">
              {analysis.comparison.higher_engagement_platform &&
                analysis.comparison.higher_engagement_platform !== "tie" && (
                  <div className="summaryMetric">
                    <dt>Engagement Winner</dt>
                    <dd>Video {analysis.comparison.higher_engagement_platform}</dd>
                  </div>
                )}
              {analysis.comparison.engagement_rate_gap !== undefined &&
                analysis.comparison.engagement_rate_gap !== null && (
                  <div className="summaryMetric">
                    <dt>Platform Gap</dt>
                    <dd>+{analysis.comparison.engagement_rate_gap.toFixed(2)}%</dd>
                  </div>
                )}
              {analysis.comparison.duration_difference_seconds !== undefined &&
                analysis.comparison.duration_difference_seconds !== null && (
                  <div className="summaryMetric">
                    <dt>Length Delta</dt>
                    <dd>{formatDuration(analysis.comparison.duration_difference_seconds)}</dd>
                  </div>
                )}
              {analysis.chunks_indexed !== undefined && analysis.chunks_indexed !== null && (
                <div className="summaryMetric">
                  <dt>Database Index</dt>
                  <dd>{analysis.chunks_indexed} RAG Chunks</dd>
                </div>
              )}
            </div>
          </article>
        )}

        <section className="comparisonGrid">
          <VideoCard label="Video A" video={analysis?.video_a} isLoading={analyzing} platform={analysis?.video_a.platform ?? platform} />
          <VideoCard label="Video B" video={analysis?.video_b} isLoading={analyzing} platform={analysis?.video_b.platform ?? platform} />
        </section>
      </section>

      <button
        className="chatResizeHandle"
        type="button"
        aria-label="Resize chat panel"
        onPointerDown={handleChatResizeStart}
      />

      <aside className="chatPanel">
        <div className="chatHeader">
          <div>
            <p className="kicker">RAG Chat</p>
            <h2>Ask about the comparison</h2>
          </div>
          <MessageSquare size={22} />
        </div>

        <div className="messages">
          {messages.length === 0 ? (
            <div className="emptyChat">
              <p style={{ margin: "0 0 4px", color: "var(--muted)", fontSize: "14px", lineHeight: 1.5 }}>
                {analysis
                  ? "Ask anything about the videos — their content, performance, hooks, or how to improve."
                  : "Analyse two videos first, then start chatting."}
              </p>
              {analysis && (
                <div className="suggestedQuestions">
                  <span className="suggestedQuestionsLabel">Try asking</span>
                  <div className="suggestedChips">
                    {SUGGESTED_QUESTIONS.map((q) => (
                      <button
                        key={q}
                        className="questionChip"
                        disabled={streaming}
                        onClick={() => setChatInput(q)}
                        type="button"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            messages.map((message, index) => {
              const isLast = index === messages.length - 1;
              const hasNoTextYet = message.role === "assistant" && !message.text;

              return (
                <article className={`message ${message.role}`} key={index}>
                    {message.role === "assistant" ? (
                      <>
                        <div className="markdownMessage">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.text}
                          </ReactMarkdown>
                          {hasNoTextYet && <span className="typingDot">●</span>}
                          {streaming && isLast && !hasNoTextYet && (
                            <span className="typingCursor">|</span>
                          )}
                        </div>
                        {!streaming && message.text && (
                          <SourceBadge text={message.text} hasCitations={Boolean(message.citations?.length)} />
                        )}
                      </>
                    ) : (
                      <p>
                        {message.text}
                      </p>
                    )}
                  {message.citations?.length ? (
                    <div className="citations">
                      {message.citations.map((citation) => (
                        <details key={citation.chunk_id} className="citationDetails">
                          <summary className="citationSummary">
                            {citation.citation}
                            {citation.start_time != null && citation.end_time != null && (
                              <span className="citationTimestamp">
                                {formatTimestamp(citation.start_time)}–{formatTimestamp(citation.end_time)}
                              </span>
                            )}
                          </summary>
                          <div className="citationContent">
                            <p className="citationTitle">{citation.title || citation.video_id}</p>
                            <blockquote className="citationQuote">{citation.chunk_text}</blockquote>
                          </div>
                        </details>
                      ))}
                    </div>
                  ) : null}
                </article>
              );
            })
          )}
        </div>

        <form className="chatForm" onSubmit={handleChat}>
          <input
            value={chatInput}
            onChange={(event) => setChatInput(event.target.value)}
            placeholder={analysis ? "Ask a follow-up..." : "Analyze videos first"}
            disabled={!analysis || streaming}
          />
          <button disabled={!chatReady || !chatInput.trim()} aria-label="Send message">
            {streaming ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
          </button>
        </form>
      </aside>
    </main>
  );
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function VideoCard({
  label,
  video,
  isLoading,
  platform,
}: {
  label: "Video A" | "Video B";
  video?: VideoAnalysis;
  isLoading: boolean;
  platform: string;
}) {
  const platformLabel = platform === "youtube" ? "YouTube Video" : "Instagram Reel";
  const rate = formatPercent(video?.engagement_rate);
  const isSilentPlaceholder =
    video?.transcript?.length === 1 &&
    video.transcript[0].text.includes("No speech detected");

  if (isLoading) {
    return (
      <article className="videoCard skeleton">
        <div className="cardHead">
          <div>
            <div className="skeletonPulse kickerPulse" />
            <div className="skeletonPulse titlePulse" />
          </div>
          <div className="skeletonPulse iconPulse" />
        </div>
        <dl className="metaGrid">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="metaGridItem skeleton">
              <div className="skeletonPulse dtPulse" />
              <div className="skeletonPulse ddPulse" />
            </div>
          ))}
        </dl>
        <div className="engagement skeleton">
          <div className="skeletonPulse rateLabelPulse" />
          <div className="skeletonPulse ratePulse" />
        </div>
      </article>
    );
  }

  return (
    <article className="videoCard animated-fade-in">
      <div className="cardHead">
        <div>
          <p className="kicker">{platformLabel} {label.slice(-1)}</p>
          <h2>{video?.metadata.title || platformLabel}</h2>
        </div>
        <Clapperboard size={24} />
      </div>

      <dl className="metaGrid">
        <Metric label="Creator" value={video?.metadata.creator || "Not loaded"} />
        <Metric label="Followers" value={formatNumber(video?.metadata.follower_count)} />
        <Metric label={platform === "instagram" ? "Views / Plays" : "Views"} value={formatNumber(video?.metadata.views)} />
        <Metric label="Likes" value={formatNumber(video?.metadata.likes)} />
        <Metric label="Comments" value={formatNumber(video?.metadata.comments)} />
        <Metric label="Duration" value={formatDuration(video?.metadata.duration)} />
        <Metric label="Uploaded" value={formatDate(video?.metadata.upload_date)} />
        <Metric label="Hashtags" value={formatHashtags(video?.metadata.hashtags)} wide />
      </dl>

      <div className="engagement">
        <span>Engagement rate</span>
        <strong>{rate}</strong>
      </div>

      <div className="cardFooter">
        {video?.url ? (
          <a href={video.url} target="_blank" rel="noreferrer" className="sourceLink">
            Open source <ArrowUpRight size={16} />
          </a>
        ) : null}
        
        {isSilentPlaceholder && (
          <span className="silentTag">
            <VolumeX size={13} /> Music Only / Silent
          </span>
        )}
      </div>
    </article>
  );
}

function Metric({ label, value, wide = false }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={`metaGridItem ${wide ? "wide" : ""}`}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function updateLastAssistant(messages: ChatMessage[], token: string): ChatMessage[] {
  return messages.map((message, index) => {
    if (index !== messages.length - 1 || message.role !== "assistant") return message;
    return { ...message, text: message.text + token };
  });
}

function attachCitations(messages: ChatMessage[], citations: Citation[]): ChatMessage[] {
  return messages.map((message, index) => {
    if (index !== messages.length - 1 || message.role !== "assistant") return message;
    return { ...message, citations };
  });
}

function replaceLastAssistant(messages: ChatMessage[], text: string): ChatMessage[] {
  return messages.map((message, index) => {
    if (index !== messages.length - 1 || message.role !== "assistant") return message;
    return { ...message, text, citations: [] };
  });
}

async function readSseStream(
  body: ReadableStream<Uint8Array>,
  handlers: {
    onConversation: (id: string) => void;
    onToken: (token: string) => void;
    onCitations: (citations: Citation[]) => void;
    onError: (message: string) => void;
  },
) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const event of events) {
      handleSseEvent(event, handlers);
    }
  }
}

function handleSseEvent(
  rawEvent: string,
  handlers: {
    onConversation: (id: string) => void;
    onToken: (token: string) => void;
    onCitations: (citations: Citation[]) => void;
    onError: (message: string) => void;
  },
) {
  const eventName = rawEvent.match(/^event: (.+)$/m)?.[1];
  const dataLine = rawEvent.match(/^data: (.+)$/m)?.[1];
  if (!eventName || !dataLine) return;

  const data = JSON.parse(dataLine);
  if (eventName === "conversation") handlers.onConversation(data);
  if (eventName === "token") handlers.onToken(data);
  if (eventName === "citations") handlers.onCitations(data);
  if (eventName === "error") handlers.onError(data);
}

async function readApiError(response: Response) {
  try {
    const data = await response.json();
    return data.detail || "Request failed.";
  } catch {
    return "Request failed.";
  }
}

function formatNumber(value?: number | null) {
  if (value === null || value === undefined) return "Not available";
  return new Intl.NumberFormat("en", { notation: "compact" }).format(value);
}

function formatPercent(value?: number | null) {
  if (value === null || value === undefined) return "Not available";
  return `${value.toFixed(2)}%`;
}

function formatDuration(value?: number | null) {
  if (!value) return "Not available";
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const seconds = value % 60;
  const secondsText = Number.isInteger(seconds)
    ? String(seconds).padStart(2, "0")
    : seconds.toFixed(3).replace(/0+$/, "").replace(/\.$/, "").padStart(2, "0");
  if (hours > 0) {
    return `${hours} hr ${minutes} min ${secondsText} sec`;
  }
  return `${minutes} min ${secondsText} sec`;
}

function formatDate(value?: string | null) {
  if (!value) return "Not available";
  if (/^\d{8}$/.test(value)) {
    return `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6)}`;
  }
  return value;
}

function formatHashtags(value?: string[] | null) {
  if (!value?.length) return "Not available";
  return value.slice(0, 8).join(" ");
}

function formatTimestamp(seconds: number): string {
  const total = Math.floor(seconds);
  const mm = Math.floor(total / 60);
  const ss = total % 60;
  return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

const SUGGESTED_QUESTIONS = [
  "Compare the hooks in the first 5 seconds.",
  "What are the engagement rates of each video?",
  "Why did one video get more engagement than the other?",
  "Suggest improvements for the lower-performing video based on what worked in the other.",
];

function SourceBadge({
  text,
  hasCitations,
}: {
  text: string;
  hasCitations: boolean;
}) {
  const lower = text.toLowerCase();
  const isInference =
    lower.includes("likely") ||
    lower.includes("may indicate") ||
    lower.includes("could suggest") ||
    lower.includes("based on available data") ||
    lower.includes("it appears");

  if (!hasCitations && !isInference) {
    return (
      <span className="sourceBadge metadata">
        🔢 Metadata
      </span>
    );
  }

  if (isInference) {
    return (
      <span className="sourceBadge inference">
        💡 Includes inference
      </span>
    );
  }

  return (
    <span className="sourceBadge transcript">
      📄 Transcript cited
    </span>
  );
}
