/* PaletteTheme — turn a word into a full set of CSS design tokens.
 *
 *   const T = await PaletteTheme.init({ base: "./palette" });
 *   T.apply("樱花");                       // sets --pn-* on :root
 *   T.tokens("樱花", { mode: "dark" });    // -> {bg, text, primary, ...}
 *
 * Depends on runtime.js (PaletteNet). No UI.
 */
(function (g) {
  "use strict";

  /* ---------- colour maths ---------- */
  function hex2rgb(h) {
    h = h.replace("#", "");
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  function rgb2hex(r, gg, b) {
    return "#" + [r, gg, b].map(v => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0")).join("");
  }
  const lin = c => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  const unlin = c => 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
  function lum(hex) {
    const [r, gg, b] = hex2rgb(hex).map(lin);
    return 0.2126 * r + 0.7152 * gg + 0.0722 * b;
  }
  function contrast(a, b) {
    const la = lum(a), lb = lum(b);
    return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
  }
  function rgb2lab(rgb) {
    const [r, gg, b] = rgb.map(lin);
    let X = (0.4124 * r + 0.3576 * gg + 0.1805 * b) / 0.95047;
    let Y = 0.2126 * r + 0.7152 * gg + 0.0722 * b;
    let Z = (0.0193 * r + 0.1192 * gg + 0.9505 * b) / 1.08883;
    const f = t => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
    X = f(X); Y = f(Y); Z = f(Z);
    return [116 * Y - 16, 500 * (X - Y), 200 * (Y - Z)];
  }
  function lab2rgb(lab) {
    const [L, a, b] = lab;
    const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
    const finv = t => t ** 3 > 0.008856 ? t ** 3 : (t - 16 / 116) / 7.787;
    const xyz = [finv(fx) * 0.95047, finv(fy), finv(fz) * 1.08883];
    return [unlin(xyz[0] * 3.2406 + xyz[1] * -1.5372 + xyz[2] * -0.4986),
            unlin(xyz[0] * -0.9689 + xyz[1] * 1.8758 + xyz[2] * 0.0415),
            unlin(xyz[0] * 0.0557 + xyz[1] * -0.2040 + xyz[2] * 1.0570)];
  }
  const lab = h => rgb2lab(hex2rgb(h));
  const fromLab = l => { const c = lab2rgb(l); return rgb2hex(c[0], c[1], c[2]); };
  const chroma = h => { const l = lab(h); return Math.hypot(l[1], l[2]); };
  const lightness = h => lab(h)[0];
  const hueOf = h => { const l = lab(h); return (Math.atan2(l[2], l[1]) * 180 / Math.PI + 360) % 360; };
  function hueDist(a, b) { const d = Math.abs(a - b) % 360; return d > 180 ? 360 - d : d; }
  function fromLCh(L, C, hDeg) {
    const h = hDeg * Math.PI / 180;
    return fromLab([L, C * Math.cos(h), C * Math.sin(h)]);
  }
  function withL(hex, L, scaleC) {
    const l = lab(hex);
    return fromLab([L, l[1] * (scaleC === undefined ? 1 : scaleC), l[2] * (scaleC === undefined ? 1 : scaleC)]);
  }
  function mix(a, b, t) {
    const A = hex2rgb(a), B = hex2rgb(b);
    return rgb2hex(A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t, A[2] + (B[2] - A[2]) * t);
  }
  function ensureContrast(fg, bg, target, dir) {
    let out = fg, L = lightness(fg);
    for (let i = 0; i < 30 && contrast(out, bg) < target; i++) {
      L += dir * 2.5;
      if (L > 98 || L < 2) break;
      out = withL(fg, L, 1);
    }
    return out;
  }

  /* ---------- palette -> design tokens ----------
   * Every one of the six colours has a job:
   *   paper  = lightest low-chroma  -> bg / surface
   *   ink    = darkest low-chroma   -> text / border
   *   primary= most chromatic       -> buttons, links
   *   accent = farthest hue         -> highlights
   *   rest   = weighted hue average -> background tint (so the word's identity
   *            survives even when two words share a primary)
   */
  function tokens(colors, opts) {
    opts = opts || {};
    const items = colors.map(c => (typeof c === "string" ? { hex: c, p: 1 } : c));
    const hexes = items.map(c => c.hex);
    const P = items.map(c => (c.p === undefined ? 1 : c.p));

    const Ls = hexes.map(lightness), Cs = hexes.map(chroma), Hs = hexes.map(hueOf);

    /* paper / ink: prefer low chroma so a vivid accent does not become the page */
    let paper = 0, ink = 0, best = -1e9;
    for (let i = 0; i < hexes.length; i++) {
      const s = Ls[i] - 0.45 * Cs[i];
      if (s > best) { best = s; paper = i; }
    }
    best = 1e9;
    for (let i = 0; i < hexes.length; i++) {
      const s = Ls[i] + 0.45 * Cs[i];
      if (s < best) { best = s; ink = i; }
    }

    /* hue of the palette as a whole: chroma- and confidence-weighted circular mean */
    let sx = 0, sy = 0, sw = 0;
    for (let i = 0; i < hexes.length; i++) {
      const w = Math.max(Cs[i] - 6, 0) * Math.sqrt(Math.max(P[i], 1e-4));
      if (w <= 0) continue;
      const h = Hs[i] * Math.PI / 180;
      sx += Math.cos(h) * w; sy += Math.sin(h) * w; sw += w;
    }
    const tintHue = sw > 0 ? (Math.atan2(sy, sx) * 180 / Math.PI + 360) % 360 : Hs[0];

    /* mode follows the whole scale's median, not the first (often darkest) swatch:
       a bright scale with a dark dominant is still a bright page. */
    const sortedL = Ls.slice().sort((a, b) => a - b);
    const medL = sortedL.length % 2
      ? sortedL[(sortedL.length - 1) / 2]
      : (sortedL[sortedL.length / 2 - 1] + sortedL[sortedL.length / 2]) / 2;
    // a near-black achromatic palette (黑/墨/炭) is dark even though it carries white steps
    const meanC = Cs.reduce((a, b) => a + b, 0) / Cs.length;
    const veryDark = opts.forceDark === false ? false : (sortedL[0] < 15 && meanC < 14);
    const autoDark = medL < 45 || veryDark;
    let mode = opts.mode;
    if (!mode) {
      if (opts.prevMode && Math.abs(medL - 45) < 6) mode = opts.prevMode;
      else mode = autoDark ? "dark" : "light";
    }

    /* surfaces are tinted by the palette's average hue, not by primary alone */
    const bgC = mode === "dark" ? 6 : 4;
    const bgL = mode === "dark" ? 11 : 97;
    let bg = fromLCh(bgL, bgC, tintHue);
    const surface = fromLCh(mode === "dark" ? 16 : 99.3, mode === "dark" ? 5 : 2.2, tintHue);
    const surfaceAlt = fromLCh(mode === "dark" ? 21 : 94.5, mode === "dark" ? 7 : 4.5, tintHue);

    /* primary / accent */
    // primary must be vivid AND among the strongly predicted colours:
    // a stray low-confidence accent must not become the brand colour.
    const pmax = Math.max.apply(null, P);
    const band = hexes.map((h, i) => i).filter(i => P[i] >= 0.35 * pmax && Cs[i] >= 8);
    const pool = band.length ? band : hexes.map((h, i) => i);
    // vivid x legible: a colour that cannot carry text on this page is not a brand colour
    // vivid x legible x predicted: dropping the probability term collapses every
    // word onto the same handful of "bright and saturated" anchors.
    const weight = i => Cs[i] * Math.sqrt(Math.max(contrast(hexes[i], bg), 1.2))
                            * Math.pow(Math.max(P[i], 1e-6) / pmax, 0.35);
    const rank = pool.slice().sort((a, b) => weight(b) - weight(a))
      .concat(hexes.map((h, i) => i).filter(i => pool.indexOf(i) < 0));
    const primaryI = (opts.pinPrimary !== undefined && opts.pinPrimary < hexes.length)
      ? opts.pinPrimary
      : (Cs[rank[0]] >= 8 ? rank[0] : paper);
    let accentI = rank.find(i => hueDist(Hs[i], Hs[primaryI]) > 32 && Cs[i] >= 10);
    if (accentI === undefined) accentI = rank[1] === undefined ? primaryI : rank[1];

    let text = withL(hexes[ink], mode === "dark" ? 94 : 15, 0.1);
    text = ensureContrast(text, bg, 9, mode === "dark" ? 1 : -1);
    let muted = mix(text, bg, 0.42);
    muted = ensureContrast(muted, bg, 4.6, mode === "dark" ? 1 : -1);
    const border = mix(bg, text, 0.16);

    let prim = withL(hexes[primaryI], mode === "dark" ? 68 : 46, 1);
    prim = ensureContrast(prim, bg, 4.6, mode === "dark" ? 1 : -1);
    const primaryHover = withL(prim, lightness(prim) + (mode === "dark" ? 6 : -6), 1);

    let acc = withL(hexes[accentI], mode === "dark" ? 70 : 50, 1);
    acc = ensureContrast(acc, bg, 3.2, mode === "dark" ? 1 : -1);

    const onPrimary = contrast("#ffffff", prim) >= contrast("#111111", prim) ? "#ffffff" : "#111111";

    return {
      mode,
      bg, surface, surfaceAlt, border,
      text, muted,
      primary: prim, primaryHover, onPrimary,
      accent: acc,
      danger: ensureContrast(withL("#c0392b", mode === "dark" ? 62 : 46, 1), bg, 3.2, mode === "dark" ? 1 : -1),
      palette: hexes,
      roles: { paper: hexes[paper], ink: hexes[ink], primary: hexes[primaryI], accent: hexes[accentI], tintHue: Math.round(tintHue) },
    };
  }

  const CSS_MAP = {
    bg: "--pn-bg", surface: "--pn-surface", surfaceAlt: "--pn-surface-alt", border: "--pn-border",
    text: "--pn-text", muted: "--pn-muted", primary: "--pn-primary", primaryHover: "--pn-primary-hover",
    onPrimary: "--pn-on-primary", accent: "--pn-accent", danger: "--pn-danger",
  };

  function css(t) {
    return Object.entries(CSS_MAP).map(([k, v]) => `${v}: ${t[k]};`).join("\n  ");
  }

  function tailwind(t) {
    const names = { bg: "bg", surface: "surface", surfaceAlt: "surface-alt", border: "border",
                    text: "text", muted: "muted", primary: "primary", primaryHover: "primary-hover",
                    onPrimary: "on-primary", accent: "accent", danger: "danger" };
    const o = {};
    for (const k of Object.keys(names)) o[names[k]] = t[k];
    return o;
  }

  /* ---------- public API ---------- */
  const cache = new Map();
  let model = null;
  const overrides = new Map();      // word / alias -> { palette, source: "table" }
  let defaults = { mix: 0.0, mode: null, base: ".", forceDark: true };
  const mq = typeof matchMedia === "function" ? matchMedia("(prefers-color-scheme: dark)") : null;

  function schemeDark() { return !!(mq && mq.matches); }

  function registerOverride(name, palette, aliases) {
    if (!name || !Array.isArray(palette) || !palette.length) return;
    const entry = { palette: palette.map(h => (typeof h === "string" ? h : h.hex)), source: "table" };
    overrides.set(name, entry);
    for (const a of aliases || []) if (a) overrides.set(a, entry);
    for (const k of [...cache.keys()]) if (k.startsWith(name + "|")) cache.delete(k);
  }

  async function loadOverrides(url) {
    try {
      const d = await fetch(url).then(r => r.json());
      const rows = d.entries || d;
      for (const [name, v] of Object.entries(rows)) {
        if (name.startsWith("_")) continue;
        const pal = Array.isArray(v) ? v : (v.palette || []);
        registerOverride(name, pal, (v && v.aliases) || []);
      }
    } catch (e) {
      console.warn("[PaletteTheme] overrides not loaded:", e.message);
    }
  }

  async function init(opts) {
    defaults = Object.assign({}, defaults, opts || {});
    if (defaults.overrides) await loadOverrides(defaults.overrides);
    model = await g.PaletteNet.load(defaults.base);
    return api;
  }

  function detectLang(w) {
    if (/[\u3040-\u30ff]/.test(w)) return "ja";
    if (/^[\x00-\x7F]+$/.test(w)) return "en";
    return "zh";
  }

  function tokensFor(word, opts) {
    if (!model) throw new Error("PaletteTheme.init() first");
    opts = opts || {};
    const mix = opts.mix === undefined ? defaults.mix : opts.mix;
    let modeOpt = opts.mode === undefined ? defaults.mode : opts.mode;
    /* mode policy:
         "system" -> follow the browser/OS colour scheme (forces dark when dark)
         "word"   -> the palette's median lightness decides
         "light" / "dark" -> literal
         null/undefined -> same as "word" */
    if (modeOpt === "system") modeOpt = schemeDark() ? "dark" : "light";
    else if (modeOpt === null || modeOpt === undefined || modeOpt === "word") modeOpt = null;
    const key = word + "|" + mix + "|" + (modeOpt || "auto") + "|" + (opts.prevMode || "-");
    if (cache.has(key)) {
      const hit = cache.get(key);
      return Object.assign({}, hit, { cached: true });
    }

    /* table first: a name added by hand must never be "interpreted" by the model */
    const ov = overrides.get(word) || overrides.get(word.trim());
    const now = typeof performance !== "undefined" && performance.now ? () => performance.now() : () => Date.now();
    let ms = 0, r;
    if (ov) {
      r = {
        colors: ov.palette.map((hex, i) => ({ hex, p: +(1 / (i + 1)).toFixed(3), score: 0, lab: null })),
        confidence: 0.99, known: true, table: true,
      };
    } else {
      const t0 = now();
      r = model.palette(word, { lang: opts.lang || detectLang(word), mix });
      ms = now() - t0;
    }
    const t = tokens(r.colors, {
      mode: modeOpt, prevMode: opts.prevMode,
      forceDark: opts.forceDark === undefined ? defaults.forceDark : opts.forceDark,
      pinPrimary: r.table ? 0 : undefined,   /* hand-written entries: first colour is the brand */
    });
    t.confidence = r.confidence;
    t.colors = r.colors;
    t.scale = (r.scale || r.colors).map(c => ({ hex: c.hex }));
    t.contrast = r.contrast ? r.contrast.hex : null;
    t.hueSpread = r.hueSpread || 0;
    t.word = word;
    t.weak = r.confidence < 0.55;          /* "this word has no strong colour" */
    t.known = r.known !== false;
    t.fromTable = !!r.table;
    t.ms = +ms.toFixed(1);          /* real forward+decode time, 0 for table hits */
    t.cached = false;
    cache.set(key, t);
    return t;
  }

  function apply(word, opts) {
    opts = opts || {};
    const root = opts.root || document.documentElement;
    const t = tokensFor(word, opts);
    for (const [k, v] of Object.entries(CSS_MAP)) root.style.setProperty(v, t[k]);
    root.setAttribute("data-pn-mode", t.mode);
    window.dispatchEvent(new CustomEvent("palettetheme", { detail: { word, tokens: t } }));
    return t;
  }

  function exportTokens(word, opts) {
    const t = tokensFor(word, opts);
    return {
      css: `:root {\n  ${css(t)}\n}`,
      tailwind: tailwind(t),
      figma: Object.fromEntries(Object.entries(CSS_MAP).map(([k]) => [k, t[k]])),
      meta: { word, mode: t.mode, confidence: t.confidence, palette: t.palette, roles: t.roles },
    };
  }

  /* re-apply when the OS/browser flips scheme */
  function onSchemeChange(cb) {
    if (!mq) return () => {};
    const h = e => { cache.clear(); cb(e.matches ? "dark" : "light"); };
    if (mq.addEventListener) mq.addEventListener("change", h);
    else if (mq.addListener) mq.addListener(h);
    return () => { if (mq.removeEventListener) mq.removeEventListener("change", h); else if (mq.removeListener) mq.removeListener(h); };
  }

  const api = {
    init, tokens: tokensFor, apply, css, exportTokens, onSchemeChange,
    registerOverride, loadOverrides,
    overrides: () => Object.fromEntries(overrides),
    scheme: () => (schemeDark() ? "dark" : "light"),
    _tokensFromPalette: tokens,
  };
  g.PaletteTheme = api;
})(typeof window !== "undefined" ? window : globalThis);
