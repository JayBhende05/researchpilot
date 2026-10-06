const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Citation = {
  chunk_id: string;
  document_id: string;
  text: string;
  score: number | null;
  metadata: { source?: string; page?: number; chunk?: number };
};

export type LatencyMs = {
  retrieval: number;
  generation: number;
  total: number;
};

export type ChatResponse = {
  question: string;
  answer: string;
  citations: Citation[];
  latency_ms: LatencyMs;
};

export async function sendChat(question: string): Promise<ChatResponse> {
  let res: Response;

  try {
    res = await fetch(`${API_URL}/api/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new Error("Cannot reach the API. Is the backend running?");
  }

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;

    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {}

    throw new Error(detail);
  }

  return res.json();
}
