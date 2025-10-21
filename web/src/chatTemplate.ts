export function formatLlamaChat(
  system: string,
  messages: { role: "user" | "assistant"; content: string }[]
) {
  const sys = system?.trim()
    ? `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n${system}\n<|eot_id|>\n`
    : "<|begin_of_text|>";
  const body = messages
    .map(
      (m) =>
        `<|start_header_id|>${m.role}<|end_header_id|>\n${m.content}\n<|eot_id|>\n`
    )
    .join("");
  const assistantCue = `<|start_header_id|>assistant<|end_header_id|>\n`;
  return sys + body + assistantCue;
}
// stop on EOS and/or "<|eot_id|>"
