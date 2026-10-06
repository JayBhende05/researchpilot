"use client";

import { FormEvent, useEffect, useRef, useState } from "react";

import { sendChat, type Citation, type LatencyMs } from "@/lib/api";
import SourcesPanel from "./SourcesPanel";

type Message =
  | { role: "user"; text: string }
  | { role: "assistant"; text: string; citations: Citation[]; latency: LatencyMs }
  | { role: "error"; text: string };

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<Citation[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();

    const question = input.trim();
    if (!question || loading) return;

    setInput("");
    setMessages((m) => [...m, { role: "user", text: question }]);
    setLoading(true);

    try {
      const res = await sendChat(question);

      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: res.answer,
          citations: res.citations,
          latency: res.latency_ms,
        },
      ]);
      setSelected(res.citations);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "error", text: err instanceof Error ? err.message : "Something went wrong" },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid h-screen grid-cols-1 md:grid-cols-[1fr_360px]">
      <main className="flex min-h-0 flex-col">
        <header className="border-b border-zinc-200 px-4 py-3 font-semibold dark:border-zinc-800">
          ResearchPilot
        </header>

        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {messages.length === 0 && (
            <p className="text-zinc-500">
              Ask a question about the indexed research papers.
            </p>
          )}

          {messages.map((m, i) => (
            <div
              key={i}
              className={m.role === "user" ? "flex justify-end" : "flex justify-start"}
            >
              <div
                className={
                  "max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-2 text-sm " +
                  (m.role === "user"
                    ? "bg-blue-600 text-white"
                    : m.role === "error"
                      ? "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-200"
                      : "bg-zinc-100 dark:bg-zinc-900")
                }
              >
                {m.text}

                {m.role === "assistant" && (
                  <div className="mt-2 flex items-center gap-3 text-xs text-zinc-500">
                    {m.citations.length > 0 && (
                      <button
                        type="button"
                        className="underline"
                        onClick={() => setSelected(m.citations)}
                      >
                        {m.citations.length} source{m.citations.length > 1 ? "s" : ""}
                      </button>
                    )}
                    <span>{(m.latency.total / 1000).toFixed(1)}s</span>
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="text-sm text-zinc-500">Thinking…</div>
          )}
          <div ref={bottomRef} />
        </div>

        <form onSubmit={onSubmit} className="flex gap-2 border-t border-zinc-200 p-4 dark:border-zinc-800">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            maxLength={1000}
            placeholder="Ask a question…"
            className="flex-1 rounded-lg border border-zinc-300 bg-transparent px-3 py-2 text-sm outline-none focus:border-blue-600 dark:border-zinc-700"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            Send
          </button>
        </form>
      </main>

      <div className="hidden min-h-0 md:block">
        <SourcesPanel citations={selected} />
      </div>
    </div>
  );
}
