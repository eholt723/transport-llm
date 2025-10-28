
import type { Retrieved } from "../../rag/types";

type Props = {
  item: Retrieved;
};

export function SourcePreview({ item }: Props) {
  const { chunk } = item;
  return (
    <div className="rounded-2xl p-4 border">
      <div className="text-base font-semibold">{chunk.title}</div>
      <div className="text-xs opacity-70 mb-2">{chunk.source} • #{chunk.offset}</div>
      <pre className="text-sm whitespace-pre-wrap">{chunk.text}</pre>
    </div>
  );
}
