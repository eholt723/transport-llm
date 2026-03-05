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
    <div className="filter-chips">
      {avail.map((d) => {
        const active = selected.map((x) => x.toLowerCase()).includes(d);
        return (
          <button
            key={d}
            className={`filter-chip${active ? " active" : ""}`}
            onClick={() => toggle(d)}
            type="button"
            title={`Filter: ${d}`}
          >
            {d}
          </button>
        );
      })}
      {avail.length === 0 && <span className="filter-empty">No domain metadata</span>}
    </div>
  );
}
