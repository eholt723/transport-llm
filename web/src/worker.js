import { MLCEngineWorkerHandler } from "@mlc-ai/web-llm";
self.onmessage = (msg) => {
  MLCEngineWorkerHandler(self, msg);
};
