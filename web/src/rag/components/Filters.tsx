import { useMemo } from "react";

type Props = {
  available: string[];
  selected: string[];
  onChange: (domains: string[]) => void;
};

export default function Filters({ available, selected, onChange }: Props) {
  const avail = useMemo(() => available.map((d) => d.toLowerCase()), [available]);

  function toggle(d: string) {
    const s = new Set(selected.map((x) => x.toLowerCase()));
    if (s.has(d)) s.delete(d); else s.add(d);
    onChange(Array.from(s));
  }

  return (
    <div className="flex flex-wrap gap-2">
      {avail.map((d) => {
        const active = selected.map((x) => x.toLowerCase()).includes(d);
        return (
          <button
            key={d}
            className={`px-3 py-1 border rounded-2xl text-sm ${active ? "bg-black text-white" : ""}`}
            onClick={() => toggle(d)}
            type="button"
            title={`Filter: ${d}`}
          >
            {d}
          </button>
        );
      })}
      {avail.length === 0 && <span className="text-sm opacity-60">No domain metadata</span>}
    </div>
  );
}
