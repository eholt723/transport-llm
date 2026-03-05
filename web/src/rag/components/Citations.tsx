import type { Retrieved } from "../../rag/types";

type Props = {
  items: Retrieved[];
  max?: number;
};

export function Citations({ items, max = 5 }: Props) {
  const shown = items.slice(0, max);
  return (
    <div className="citations-list">
      {shown.map(({ chunk, score }) => (
        <div key={chunk.id} className="citation-item">
          <div className="citation-title">
            {chunk.title}
            <span className="citation-score">{score.toFixed(3)}</span>
          </div>
          <div className="citation-meta">{chunk.source} · chunk {chunk.offset}</div>
          <div className="citation-text">{chunk.text}</div>
        </div>
      ))}
    </div>
  );
}
