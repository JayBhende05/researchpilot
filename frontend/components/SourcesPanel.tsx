import type { Citation } from "@/lib/api";

export default function SourcesPanel({ citations }: { citations: Citation[] }) {
  return (
    <aside className="flex h-full flex-col gap-3 overflow-y-auto border-l border-zinc-200 p-4 dark:border-zinc-800">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-500">
        Sources
      </h2>

      {citations.length === 0 && (
        <p className="text-sm text-zinc-500">
          Sources cited by the answer appear here.
        </p>
      )}

      {citations.map((c, i) => (
        <details
          key={c.chunk_id}
          className="rounded-lg border border-zinc-200 p-3 text-sm dark:border-zinc-800"
        >
          <summary className="cursor-pointer font-medium">
            [{i + 1}] {c.document_id}
            {c.metadata.page !== undefined && ` · p.${c.metadata.page}`}
          </summary>
          <p className="mt-2 whitespace-pre-wrap text-zinc-600 dark:text-zinc-400">
            {c.text}
          </p>
          {c.score !== null && (
            <p className="mt-2 text-xs text-zinc-500">
              relevance score: {c.score.toFixed(2)}
            </p>
          )}
        </details>
      ))}
    </aside>
  );
}
