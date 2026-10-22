/**
 * cc-switch-sync
 *
 * Syncs the *currently active* cc-switch providers into pi at startup.
 * The pi provider names are stable and vendor-agnostic:
 *   - "cc-claude" : whatever cc-switch's active `claude` provider is
 *   - "cc-codex"  : whatever cc-switch's active `codex` provider is
 *
 * Switch a provider in cc-switch, then restart pi (or /reload) to pick it up.
 *
 * Sources:
 *   - model list      : discovered live from <baseUrl>/v1/models (sole source)
 *   - context window   : enriched from models.dev catalog, persisted to a local
 *                        cache and re-fetched in the background on each startup
 *                        (non-blocking, silent fail). Per-startup enrichment
 *                        uses the *previous* run's cache; first run uses
 *                        per-api defaults until the cache populates. Models the
 *                        catalog does not know fall back to CONTEXT_DEFAULTS.
 *   - reasoning + input: also enriched from the catalog (modalities.input maps
 *                        to pi's text/image inputs); unknown models assume
 *                        reasoning + text/image.
 *   - api type         : cc-switch provider meta.apiFormat / Codex wire_api
 */

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const DB = join(homedir(), ".cc-switch", "cc-switch.db");
const CACHE = join(homedir(), ".pi", "agent", "cc-switch-models-cache.json");
const MODELS_DEV_URL = "https://models.dev/api.json";
const CATALOG_TIMEOUT_MS = 15000;

// Context windows are not discoverable; sensible per-api fallbacks.
const CONTEXT_DEFAULTS = {
  "anthropic-messages": 200000,
  "openai-responses": 256000,
  "openai-completions": 256000,
};
const MAX_TOKENS_DEFAULT = 32000;
const FETCH_TIMEOUT_MS = 8000;

function sqlite(sql) {
  const out = execFileSync("sqlite3", ["-json", DB, sql], {
    encoding: "utf8",
  }).trim();
  return out ? JSON.parse(out) : [];
}

function currentProvider(appType) {
  const rows = sqlite(
    `SELECT id, name, settings_config, meta FROM providers ` +
      `WHERE app_type='${appType}' AND is_current=1 LIMIT 1;`,
  );
  if (!rows.length) return null;
  return {
    id: rows[0].id,
    name: rows[0].name,
    config: JSON.parse(rows[0].settings_config || "{}"),
    meta: JSON.parse(rows[0].meta || "{}"),
  };
}

function tomlScalar(toml, key) {
  const m = toml.match(new RegExp(`^\\s*${key}\\s*=\\s*"([^"]*)"`, "m"));
  return m ? m[1] : null;
}

async function discoverModels(baseUrl, apiKey) {
  const base = baseUrl.replace(/\/+$/, "");
  const url = /\/v\d+$/.test(base) ? `${base}/models` : `${base}/v1/models`;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const body = await res.json();
    return Array.isArray(body?.data) ? body.data : [];
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// models.dev catalog: persistent cache + background refresh
// ---------------------------------------------------------------------------

// Strip a trailing -YYYYMMDD so served ids (often undated) match catalog keys.
function normId(id) {
  return String(id).replace(/-\d{8}$/, "");
}

// Read the persisted catalog cache. Returns a flat map: normId -> { context, output }.
function loadCatalogCache() {
  try {
    const raw = JSON.parse(readFileSync(CACHE, "utf8"));
    return new Map(Object.entries(raw.limits || {}));
  } catch {
    return new Map();
  }
}

// Flatten the models.dev payload into normId -> { context, output, reasoning, input }.
function flattenCatalog(payload) {
  const limits = {};
  for (const provider of Object.values(payload || {})) {
    const models = provider?.models || {};
    for (const [key, model] of Object.entries(models)) {
      const lim = model?.limit || {};
      if (!Number(lim.context)) continue;
      // pi only supports text/image inputs; map and filter the catalog list.
      const catalogInput = model?.modalities?.input || [];
      const input = catalogInput.filter((t) => t === "text" || t === "image");
      const entry = {
        context: Number(lim.context) || 0,
        output: Number(lim.output) || 0,
        reasoning: Boolean(model?.reasoning),
        input: input.length ? input : ["text"],
      };
      for (const k of new Set([key, normId(key), model?.id, normId(model?.id || "")])) {
        if (k && limits[k] === undefined) limits[k] = entry;
      }
    }
  }
  return limits;
}

// Fire-and-forget refresh: fetch models.dev, persist to cache. Never throws,
// never awaited by startup. Refreshes the cache for the *next* launch.
async function refreshCatalogInBackground() {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), CATALOG_TIMEOUT_MS);
  try {
    const res = await fetch(MODELS_DEV_URL, { signal: ctrl.signal });
    if (!res.ok) return;
    const payload = await res.json();
    const limits = flattenCatalog(payload);
    if (Object.keys(limits).length === 0) return;
    writeFileSync(
      CACHE,
      JSON.stringify({ fetchedAt: new Date().toISOString(), limits }),
      { mode: 0o600 },
    );
  } catch {
    // silent: stale cache or defaults still drive enrichment
  } finally {
    clearTimeout(timer);
  }
}

// Build pi model definitions from the discovered list, enriching context/output
// limits, reasoning, and input modalities from the persisted catalog cache when
// known, else per-api defaults and permissive assumptions.
function buildModels(rawModels, api, catalog) {
  const fallbackContext = CONTEXT_DEFAULTS[api] ?? 200000;
  return rawModels
    .filter((m) => m?.id && !/image|embed|tts|whisper|audio/i.test(m.id))
    .map((m) => {
      const known = catalog.get(m.id) ?? catalog.get(normId(m.id));
      return {
        id: m.id,
        name: m.id,
        reasoning: known ? known.reasoning : true,
        input: known?.input?.length ? known.input : ["text", "image"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: known?.context || fallbackContext,
        maxTokens: known?.output || MAX_TOKENS_DEFAULT,
      };
    });
}

async function syncClaude(pi, catalog) {
  const prov = currentProvider("claude");
  if (!prov) return;
  const apiKey = prov.config?.env?.ANTHROPIC_AUTH_TOKEN;
  const baseUrl = prov.config?.env?.ANTHROPIC_BASE_URL;
  if (!apiKey || !baseUrl) return;

  const api =
    prov.meta?.apiFormat === "openai" ? "openai-completions" : "anthropic-messages";

  let models = [];
  try {
    models = buildModels(await discoverModels(baseUrl, apiKey), api, catalog);
  } catch (e) {
    console.error(`[cc-switch-sync] claude model discovery failed: ${e.message}`);
  }
  if (!models.length) {
    console.error("[cc-switch-sync] no claude models resolved; skipping");
    return;
  }

  pi.registerProvider("cc-claude", {
    name: `cc-switch: ${prov.name}`,
    baseUrl,
    apiKey,
    api,
    models,
  });
}

async function syncCodex(pi, catalog) {
  const prov = currentProvider("codex");
  if (!prov) return;
  const apiKey = prov.config?.auth?.OPENAI_API_KEY;
  const toml = prov.config?.config ?? "";
  const baseUrl = tomlScalar(toml, "base_url");
  const configuredModel = tomlScalar(toml, "model");
  const wireApi = tomlScalar(toml, "wire_api");
  if (!apiKey || !baseUrl) return;

  const api = wireApi === "chat" ? "openai-completions" : "openai-responses";

  let models = [];
  try {
    models = buildModels(await discoverModels(baseUrl, apiKey), api, catalog);
  } catch (e) {
    console.error(`[cc-switch-sync] codex model discovery failed: ${e.message}`);
  }
  // Fall back to the model named in the cc-switch TOML config.
  if (!models.length && configuredModel) {
    models = buildModels([{ id: configuredModel }], api, catalog);
  }
  if (!models.length) {
    console.error("[cc-switch-sync] no codex models resolved; skipping");
    return;
  }

  pi.registerProvider("cc-codex", {
    name: `cc-switch: ${prov.name}`,
    baseUrl,
    apiKey,
    api,
    models,
  });
}

export default async function (pi) {
  // Enrich from the previous run's persisted catalog (instant, offline-safe).
  const catalog = loadCatalogCache();

  // Register providers using the cached catalog.
  await Promise.allSettled([syncClaude(pi, catalog), syncCodex(pi, catalog)]);

  // Refresh the catalog for the next startup. Not awaited: cannot block or
  // delay pi startup, and any failure is swallowed.
  void refreshCatalogInBackground();
}
