// scripts/make-404.js
import { copyFileSync, mkdirSync } from "fs";
mkdirSync("dist", { recursive: true });
copyFileSync("dist/index.html", "dist/404.html");
console.log("Created dist/404.html for SPA fallback");
