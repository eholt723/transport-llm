
import type { Retrieved } from "../../rag/types";

type Props = {
  items: Retrieved[];
  max?: number;
};

export function Citations({ items, max = 5 }: Props) {
  const shown = items.slice(0, max);
  return (
    <div className="mt-3 space-y-2">
      {shown.map(({ chunk, score }) => (
        <div key={chunk.id} className="rounded-2xl p-3 border">
          <div className="text-sm font-semibold">
            {chunk.title} <span className="opacity-60">({score.toFixed(3)})</span>
          </div>
          <div className="text-xs opacity-70">{chunk.source} • #{chunk.offset}</div>
          <div className="text-sm mt-2 line-clamp-4 whitespace-pre-wrap">{chunk.text}</div>
        </div>
      ))}
    </div>
  );
}
