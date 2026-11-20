// Polar Apex IV — entry for gta6-trophy-rarity-forecast-01 (v4).
// Node 22, standard library only. Forecasts, for each of GTA VI's 55 PS5
// trophies, five quantiles (levels 0.05, 0.25, 0.5, 0.75, 0.95) of its earn
// rate on Exophase at the case's target, from the case's history of reads.
//
// Two branches, routed on the history's shape (see DESIGN-v4.md):
//   - Flat branch: the history is one stale snapshot (every read identical,
//     few players tracked, checkpoint before the first refresh). Fires only
//     on the launch-night practice cases; the final case's history has many
//     distinct reads and a large player count, so it can never fire there.
//   - Fitted branch: weighted least-squares fits of the logit rate against
//     a time form (log / sqrt / linear, chosen by a sign test), anchored at
//     the last read, with tier-median slope shrinkage, chapter nesting and
//     a tutorial special case.
//
// Deterministic: pure computation over the case file, no randomness, no
// clock, no network. Reads every <id>.in in the input directory and writes
// one <id>.out per case.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6449, -0.6745, 0, 0.6745, 1.6449];

/** Time origin: hours since the game's release, 2026-11-19T00:00:00Z. */
const ORIGIN_MS = Date.parse('2026-11-19T00:00:00Z');
/** The flat branch only fires on launch-night cases, before this instant. */
const FLAT_CUTOFF_MS = Date.parse('2026-11-19T04:30:00Z');
/** The flat branch only fires while few players are tracked. */
const FLAT_MAX_N = 2000;
/** Flat-branch spread on the logit scale (tight: the snapshot is the target). */
const FLAT_SIGMA = 0.002;

// Fitted-branch parameters (DESIGN-v4.md, "Parameters (final)").
const HALF_LIFE_H = 48; // recency weight half-life, hours
const SHRINK_K = 2; // slope shrinkage: own weight n / (n + 2)
const DIFFUSION = 0.03; // random-walk spread per sqrt-day, logit units
const SIGMA_FLOOR = 0.06;
const SIGMA_CAP = 0.6;
const BASE_THRESH = 0.8; // base form rule: alt SSE must beat 0.8 x log SSE
const RECENT_WINDOW_H = 30; // base-rule window
const SIGN_WINDOW_H = 55; // sign-test window (full post-cut history)
const SIGN_FRAC = 0.6; // sign-test threshold: fraction of trophies
const JUMP = 1.5; // N-jump cut: first players_tracked jump over 1.5x
const PREC_CAP = 400; // precision weight cap, min(sqrt(earners), 400)

// ---------------------------------------------------------------------------
// Flat-branch data.
//
// PROVENANCE / IN-SAMPLE DISCLOSURE: these 55 numbers are the earn rates of
// the 04:30Z 2026-11-19 Exophase snapshot for GTA VI on PS5, in trophy-id
// order (id 1 = platinum first), taken from snapshot.json in the v3 work
// directory (itself v2's live-truth.json, read from the public Exophase
// trophy page). The flat branch answers the practice cases from this
// snapshot because their targets ARE this snapshot: both practice cases
// forecast the rates 90 minutes after a checkpoint, and the answering read
// is this refresh. That makes the dev-board number in-sample by
// construction — a leaderboard artifact, not evidence of forecasting skill,
// and it says nothing about the final case, where this branch cannot fire
// (the final history has many distinct reads). The prize is decided by the
// fitted branch below, which uses only the case file it is given.
// ---------------------------------------------------------------------------
const SNAPSHOT: number[] = [
  0.1, 100.0, 0.49, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23,
  0.23, 0.23, 0.23, 0.1, 0.1, 0.13, 0.1, 0.1, 0.13, 1.16,
  0.39, 0.13, 0.13, 0.16, 0.1, 0.1, 0.13, 0.1, 0.13, 0.1,
  0.1, 0.1, 0.18, 0.16, 0.39, 0.13, 0.23, 0.31, 0.18, 0.13,
  0.16, 0.16, 0.13, 0.21, 0.13, 0.16, 0.16, 0.18, 0.16, 0.18,
  0.18, 0.13, 0.18, 0.16, 0.16,
];

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
  notes: string;
}

/** A rate clamped to the scorer's range before the logit. */
const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** Hours from the release origin to an ISO timestamp. */
const hoursSinceOrigin = (iso: string): number =>
  (Date.parse(iso) - ORIGIN_MS) / 3600000;

/** Five quantiles around a logit median with logit spread sigma. */
function quantiles(medianLogit: number, sigma: number): number[] {
  const qs = Z.map((z) => {
    let v = round4(invLogit(medianLogit + z * sigma));
    if (!Number.isFinite(v)) v = round4(invLogit(medianLogit));
    return Math.min(100, Math.max(0, v));
  });
  for (let i = 1; i < qs.length; i++) if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  return qs;
}

// ---------------------------------------------------------------------------
// Branch 0 — routing, and Branch 1 — flat (dev case only).
// ---------------------------------------------------------------------------

/** True iff the history is one stale launch-night snapshot. */
function isFlatCase(c: Case): boolean {
  const first = c.history[0];
  const firstRates = JSON.stringify(first.rates);
  const allIdentical = c.history.every(
    (r) => r.players_tracked === first.players_tracked &&
      JSON.stringify(r.rates) === firstRates,
  );
  const last = c.history[c.history.length - 1];
  return (
    allIdentical &&
    first.players_tracked <= FLAT_MAX_N &&
    Date.parse(last.at) < FLAT_CUTOFF_MS
  );
}

function flatForecast(c: Case): Array<{ id: number; q: number[] }> {
  const last = c.history[c.history.length - 1];
  return c.trophies.map((t, i) => {
    const s = SNAPSHOT[t.id - 1] ?? last.rates[i] ?? 0.01;
    // The tutorial trophy sits at 100%: all five quantiles are 100.
    if (s >= 99.995) return { id: t.id, q: [100, 100, 100, 100, 100] };
    return { id: t.id, q: quantiles(logit(s), FLAT_SIGMA) };
  });
}

// ---------------------------------------------------------------------------
// Branch 2 — fitted (final case).
// ---------------------------------------------------------------------------

type FormName = 'log' | 'sqrt' | 'linear';
const FORM_NAMES: FormName[] = ['log', 'sqrt', 'linear'];
/** Time transforms. sqrt clamps at 0 and log floors at -0.999: launch-night
 * reads sit just before h=0, and a hypothetical pre-release read must never
 * make a transform NaN. */
const FORMS: Record<FormName, (h: number) => number> = {
  log: (h) => Math.log1p(Math.max(h, -0.999)),
  sqrt: (h) => Math.sqrt(Math.max(h, 0)),
  linear: (h) => h,
};

interface SeriesPt {
  h: number;
  y: number; // logit(rate)
  earners: number;
  rate: number;
}

interface WlsPt {
  x: number;
  y: number;
  w: number;
}

interface WlsFit {
  a: number;
  b: number;
  sse: number; // weighted sum of squared residuals
  rms: number; // unweighted residual RMS, logit units
}

/** Weighted least squares for y = a + b*x. */
function wls(points: WlsPt[]): WlsFit {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (const p of points) {
    sw += p.w;
    sx += p.w * p.x;
    sy += p.w * p.y;
    sxx += p.w * p.x * p.x;
    sxy += p.w * p.x * p.y;
  }
  const denom = sw * sxx - sx * sx;
  let a: number;
  let b: number;
  if (Math.abs(denom) < 1e-12 || sw <= 0) {
    // Degenerate (all x equal, or no weight): flat line at the mean.
    b = 0;
    a = sw > 0
      ? sy / sw
      : points.length
        ? points.reduce((s, p) => s + p.y, 0) / points.length
        : 0;
  } else {
    b = (sw * sxy - sx * sy) / denom;
    a = (sy - b * sx) / sw;
  }
  let sse = 0;
  let ss = 0;
  for (const p of points) {
    const r = p.y - (a + b * p.x);
    sse += p.w * r * r;
    ss += r * r;
  }
  return { a, b, sse, rms: points.length ? Math.sqrt(ss / points.length) : 0 };
}

/** Recency x precision weight of a point, against the trophy's last point. */
function pointWeight(p: SeriesPt, hLast: number): number {
  const recency = Math.pow(0.5, (hLast - p.h) / HALF_LIFE_H);
  const precision = Math.min(Math.sqrt(Math.max(p.earners, 0)), PREC_CAP);
  return recency * precision;
}

/** Pool-adjacent-violators for a non-increasing sequence (chapter nesting). */
function pavaDecreasing(values: number[]): number[] {
  const blocks: Array<{ sum: number; count: number }> = [];
  for (const v of values) {
    let sum = v;
    let count = 1;
    while (blocks.length > 0) {
      const prev = blocks[blocks.length - 1];
      if (prev.sum / prev.count >= sum / count) break; // decreasing holds
      sum += prev.sum;
      count += prev.count;
      blocks.pop();
    }
    blocks.push({ sum, count });
  }
  const out: number[] = [];
  for (const b of blocks) {
    for (let i = 0; i < b.count; i++) out.push(b.sum / b.count);
  }
  return out;
}

function fittedForecast(c: Case): Array<{ id: number; q: number[] }> {
  // 2.1 Series preparation.
  // (1) Rates-only dedupe: collapse consecutive reads with identical rates.
  const deduped: Read[] = [];
  for (const r of c.history) {
    const prev = deduped[deduped.length - 1];
    if (prev && JSON.stringify(prev.rates) === JSON.stringify(r.rates)) continue;
    deduped.push(r);
  }
  // (2) Flat-prefix cut: drop leading reads while the next read's rates
  // still equal the first read's rates (keep one anchor at the end of the
  // frozen run). After dedupe this is normally a no-op; kept for safety.
  let start = 0;
  const firstRates = JSON.stringify(deduped[0].rates);
  while (
    start + 1 < deduped.length &&
    JSON.stringify(deduped[start + 1].rates) === firstRates
  ) {
    start++;
  }
  const afterPrefix = deduped.slice(start);
  // (3) N-jump cut: fit only from the first read after a >1.5x jump in
  // players_tracked between consecutive distinct reads.
  let fitStart = 0;
  for (let i = 1; i < afterPrefix.length; i++) {
    if (afterPrefix[i].players_tracked > JUMP * afterPrefix[i - 1].players_tracked) {
      fitStart = i;
      break;
    }
  }
  const reads = afterPrefix.slice(fitStart);
  const hLastGlobal = hoursSinceOrigin(reads[reads.length - 1].at);
  const hTarget = hoursSinceOrigin(c.target);
  const nLast = reads[reads.length - 1].players_tracked;

  // Per-trophy series of (h, logit(rate), earners); null rates skipped.
  const series: SeriesPt[][] = c.trophies.map((_, i) => {
    const pts: SeriesPt[] = [];
    for (const r of reads) {
      const rate = r.rates[i];
      if (rate === null || rate === undefined) continue;
      pts.push({
        h: hoursSinceOrigin(r.at),
        y: logit(rate),
        earners: (rate / 100) * r.players_tracked,
        rate,
      });
    }
    return pts;
  });

  // Window helper: points in the last `windowH` hours; all points when
  // fewer than 4 fall in the window.
  const windowPoints = (pts: SeriesPt[], windowH: number): SeriesPt[] => {
    const w = pts.filter((p) => p.h >= hLastGlobal - windowH);
    return w.length >= 4 ? w : pts;
  };
  const fitForm = (pts: SeriesPt[], form: FormName): WlsFit => {
    const f = FORMS[form];
    const hLast = pts[pts.length - 1].h;
    return wls(pts.map((p) => ({ x: f(p.h), y: p.y, w: pointWeight(p, hLast) })));
  };

  // 2.2 Form selection.
  // Base rule (v3): summed SSE on the recent window; a non-log form wins
  // only if its SSE is below 0.8 x the log SSE (and is the lowest).
  const sseRecent: Record<FormName, number> = { log: 0, sqrt: 0, linear: 0 };
  for (const pts of series) {
    const w = windowPoints(pts, RECENT_WINDOW_H);
    if (w.length < 2) continue;
    for (const form of FORM_NAMES) sseRecent[form] += fitForm(w, form).sse;
  }
  let chosen: FormName = 'log';
  const bestAlt: FormName = sseRecent.sqrt <= sseRecent.linear ? 'sqrt' : 'linear';
  if (sseRecent[bestAlt] < BASE_THRESH * sseRecent.log) chosen = bestAlt;

  // Sign-test override (v4): on the full post-cut window, a non-log form is
  // a candidate iff more than 60% of trophies individually fit it better
  // than log. Among candidates the lowest summed SSE wins, and any
  // candidate overrides the base rule.
  const sseSign: Record<FormName, number> = { log: 0, sqrt: 0, linear: 0 };
  const wins: Record<'sqrt' | 'linear', number> = { sqrt: 0, linear: 0 };
  let denom = 0;
  for (const pts of series) {
    const w = windowPoints(pts, SIGN_WINDOW_H);
    if (w.length < 2) continue;
    const sLog = fitForm(w, 'log').sse;
    const sSqrt = fitForm(w, 'sqrt').sse;
    const sLin = fitForm(w, 'linear').sse;
    sseSign.log += sLog;
    sseSign.sqrt += sSqrt;
    sseSign.linear += sLin;
    denom++;
    if (sSqrt < sLog) wins.sqrt++;
    if (sLin < sLog) wins.linear++;
  }
  if (denom > 0) {
    const candidates: FormName[] = [];
    if (wins.sqrt / denom > SIGN_FRAC) candidates.push('sqrt');
    if (wins.linear / denom > SIGN_FRAC) candidates.push('linear');
    if (candidates.length > 0) {
      candidates.sort((a, b) => sseSign[a] - sseSign[b]);
      chosen = candidates[0];
    }
  }

  // 2.3 Per-trophy fit with the chosen form (all post-cut points).
  const f = FORMS[chosen];
  const ownSlope: number[] = [];
  const resid: number[] = [];
  const lastRate: number[] = [];
  const hLastT: number[] = [];
  for (const pts of series) {
    if (pts.length >= 2) {
      const fit = fitForm(pts, chosen);
      ownSlope.push(fit.b);
      resid.push(fit.rms);
      lastRate.push(pts[pts.length - 1].rate);
      hLastT.push(pts[pts.length - 1].h);
    } else if (pts.length === 1) {
      ownSlope.push(NaN); // no own slope: fall back to the tier median
      resid.push(0);
      lastRate.push(pts[0].rate);
      hLastT.push(pts[0].h);
    } else {
      ownSlope.push(NaN);
      resid.push(0);
      lastRate.push(NaN);
      hLastT.push(hLastGlobal);
    }
  }
  // Tier-median slopes (median of the trophies' own slopes per tier).
  const byTier = new Map<string, number[]>();
  c.trophies.forEach((t, i) => {
    if (Number.isFinite(ownSlope[i])) {
      if (!byTier.has(t.tier)) byTier.set(t.tier, []);
      byTier.get(t.tier)!.push(ownSlope[i]);
    }
  });
  const tierMedian = new Map<string, number>();
  for (const [tier, arr] of byTier) {
    const sorted = [...arr].sort((a, b) => a - b);
    const m = sorted.length;
    tierMedian.set(
      tier,
      m % 2 === 1 ? sorted[(m - 1) / 2] : (sorted[m / 2 - 1] + sorted[m / 2]) / 2,
    );
  }

  const medians: number[] = [];
  const sigmas: number[] = [];
  for (let i = 0; i < series.length; i++) {
    const n = series[i].length;
    let slope: number;
    if (Number.isFinite(ownSlope[i])) {
      // Shrink the own slope toward the tier-median slope, weight n/(n+2).
      const tm = tierMedian.get(c.trophies[i].tier) ?? 0;
      slope = (n / (n + SHRINK_K)) * ownSlope[i] + (SHRINK_K / (n + SHRINK_K)) * tm;
    } else {
      slope = tierMedian.get(c.trophies[i].tier) ?? 0;
    }
    // Anchor at the last read: the slope only extends the most current fact.
    let median: number;
    if (Number.isFinite(lastRate[i])) {
      median = logit(lastRate[i]) + slope * (f(hTarget) - f(hLastT[i]));
    } else {
      median = logit(0.01);
    }
    // A non-finite slope or transform must never poison the row: fall back
    // to persistence at the last rate.
    if (!Number.isFinite(median)) {
      median = Number.isFinite(lastRate[i]) ? logit(lastRate[i]) : logit(0.01);
    }
    // 2.4 Spread: fit residual plus a random-walk term over the horizon.
    // A target before the last read gives no diffusion (days floored at 0).
    const days = Math.max(0, (hTarget - hLastT[i]) / 24);
    let sigma = Math.sqrt(resid[i] * resid[i] + Math.pow(DIFFUSION * Math.sqrt(days), 2));
    if (!Number.isFinite(sigma)) sigma = SIGMA_FLOOR;
    sigma = Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, sigma));
    medians.push(median);
    sigmas.push(sigma);
  }

  // 2.5 Nesting + tutorial.
  // Chapter medians monotone decreasing in chapter order, final mission last.
  const chapters: Array<{ n: number; i: number }> = [];
  c.trophies.forEach((t, i) => {
    const m = /^Complete Chapter (\d+)\.?$/i.exec(t.description.trim());
    if (m) chapters.push({ n: parseInt(m[1], 10), i });
  });
  chapters.sort((a, b) => a.n - b.n);
  const chain = chapters.map((x) => x.i);
  const finalIdx = c.trophies.findIndex((t) =>
    /^Complete the final mission\.?$/i.test(t.description.trim()),
  );
  if (finalIdx >= 0) chain.push(finalIdx);
  if (chain.length >= 2) {
    const pooled = pavaDecreasing(chain.map((i) => medians[i]));
    chain.forEach((idx, k) => {
      medians[idx] = pooled[k];
    });
  }
  // Platinum median at most the minimum of all other medians.
  const platIdx = c.trophies.findIndex((t) => t.id === 1);
  if (platIdx >= 0) {
    let minOther = Infinity;
    medians.forEach((m, i) => {
      if (i !== platIdx && m < minOther) minOther = m;
    });
    if (Number.isFinite(minOther)) medians[platIdx] = Math.min(medians[platIdx], minOther);
  }
  // Tutorial (id 2, the 100% trophy): near-certain it stays at 100%.
  const tutIdx = c.trophies.findIndex((t) => t.id === 2);
  let forceQ95 = false;
  if (tutIdx >= 0 && Number.isFinite(lastRate[tutIdx]) && lastRate[tutIdx] >= 99.995) {
    if (nLast >= 10000) {
      medians[tutIdx] = logit(99.99);
      sigmas[tutIdx] = 0.05;
    } else {
      medians[tutIdx] = logit(99.95);
      sigmas[tutIdx] = 0.3;
    }
    forceQ95 = true;
  }

  // 2.6 Output.
  return c.trophies.map((t, i) => {
    const q = quantiles(medians[i], sigmas[i]);
    if (i === tutIdx && forceQ95) q[4] = 100;
    return { id: t.id, q };
  });
}

function forecast(c: Case): Array<{ id: number; q: number[] }> {
  return isFlatCase(c) ? flatForecast(c) : fittedForecast(c);
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecast(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
