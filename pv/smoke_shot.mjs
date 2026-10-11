// Smoke test: render a tiny HTML page with CJK text and screenshot it with headless Chromium.
// usage: node pv/smoke_shot.mjs out/smoke_shot.png
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
process.env.PLAYWRIGHT_BROWSERS_PATH ??= path.join(root, ".cache", "ms-playwright");
const { chromium } = await import("playwright");

const out = process.argv[2] ?? path.join(root, "out", "smoke_shot.png");
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 354, height: 200 }, deviceScaleFactor: 1 });
await page.setContent(`<body style="margin:0;background:#0b1020;color:#c4d4ff;font:16px 'Microsoft YaHei',sans-serif">
  <div style="padding:16px"><b style="color:#4d6bfe">smoke test</b><br>你好，世界 · hello, world</div></body>`);
await page.evaluate(() => document.fonts.ready);
await page.screenshot({ path: out });
await browser.close();
console.log("ok", out);
