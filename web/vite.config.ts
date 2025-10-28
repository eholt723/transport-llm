import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,     // expose to Windows host
    port: 5173,     // fixed port
    strictPort: true
  }
});
