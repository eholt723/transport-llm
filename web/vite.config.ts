import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// repo base - github.com/eholt723/transport-llm
export default defineConfig({
  base: "/transport-llm/",
  plugins: [react()],
  server: { host: true, port: 5173, strictPort: true },
});
