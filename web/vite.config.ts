import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Deploy
export default defineConfig({
  plugins: [react()],
  base: "/transport-llm/",
});
