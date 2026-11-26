// PLAYWRIGHT_CORE=/path/to/playwright-core/index.mjs node scripts/capture_training_page.mjs [out.png]
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const { chromium } = await import(process.env.PLAYWRIGHT_CORE ?? "playwright-core");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const page_url = pathToFileURL(path.join(root, "docs", "dpr-training-run.html")).href;
const out = process.argv[2] ?? path.join(root, "docs", "dpr-training-run.png");

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 });
const errors = [];
page.on("console", (msg) => { if (msg.type() === "error" || msg.type() === "warning") errors.push(`${msg.type()}: ${msg.text()}`); });
page.on("pageerror", (err) => errors.push(`pageerror: ${err.message}`));
page.on("requestfailed", (req) => errors.push(`request failed: ${req.url()}`));
page.on("request", (req) => { if (!req.url().startsWith("file:")) errors.push(`network request: ${req.url()}`); });

await page.goto(page_url, { waitUntil: "load" });

// drive the controls so their handlers run
const loss = page.locator("#loss-svg");
const box = await loss.boundingBox();
await loss.hover({ position: { x: box.width * 0.4, y: box.height * 0.5 } });
const readout = await page.locator("#loss-readout").textContent();
await page.locator("#show-lr").check();
await page.locator("#show-lr").uncheck();
for (const id of ["#val-metric", "#cmp-metric", "#cmp-split", "#val2-metric", "#res-split"]) {
  const buttons = page.locator(`${id} button`);
  for (let i = (await buttons.count()) - 1; i >= 0; i--) await buttons.nth(i).click();
}
await page.mouse.move(0, 0);

await page.screenshot({ path: out, fullPage: true });
await browser.close();

console.log(`loss readout: ${readout}`);
console.log(`screenshot: ${out}`);
if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}
console.log("console: no errors");
