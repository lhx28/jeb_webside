/* PaletteNet runtime — byte-level tiny transformer -> 256 colour anchors -> hex palette.
 * No dependencies. Model: model.bin (int8) + model.json + codebook.json.
 */
(function (global) {
  "use strict";

  const PAD = 0, BOS = 1, EOS = 2, UNK = 3, CHAR_BASE = 4, WORD_BASE = 4096, NAME_BASE = 7168;
  const LANG = { zh: 0, en: 1, ja: 2 };

  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  function gelu(x) {
    return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
  }
  function layerNorm(x, g, b, eps) {
    const n = x.length;
    let m = 0;
    for (let i = 0; i < n; i++) m += x[i];
    m /= n;
    let v = 0;
    for (let i = 0; i < n; i++) { const d = x[i] - m; v += d * d; }
    v = 1 / Math.sqrt(v / n + eps);
    const y = new Float32Array(n);
    for (let i = 0; i < n; i++) y[i] = (x[i] - m) * v * g[i] + b[i];
    return y;
  }
  /* y = x @ W^T + b   (W row-major [out][in]) */
  function linear(x, W, b, out, inp) {
    const y = new Float32Array(out);
    for (let j = 0; j < out; j++) {
      const base = j * inp;
      let s = b ? b[j] : 0;
      for (let i = 0; i < inp; i++) s += x[i] * W[base + i];
      y[j] = s;
    }
    return y;
  }

  class PaletteModel {
    constructor(t) { this.t = t; this.cb = null; }

    static async load(base) {
      base = base || ".";
      const [meta, buf, cb, vocab] = await Promise.all([
        fetch(base + "/model.json").then(r => r.json()),
        fetch(base + "/model.bin").then(r => r.arrayBuffer()),
        fetch(base + "/codebook.json").then(r => r.json()),
        fetch(base + "/vocab.json").then(r => r.json()),
      ]);
      const t = {};
      for (const e of meta.tensors) {
        const slice = buf.slice(e.offset, e.offset + e.bytes);
        if (e.dtype === "int8") {
          const q = new Int8Array(slice);
          const sName = e.name.replace(/\.q$/, ".s");
          const se = meta.tensors.find(z => z.name === sName);
          if (!se) throw new Error("missing scale " + sName);
          const sc = new Float32Array(buf.slice(se.offset, se.offset + se.bytes));
          const f = new Float32Array(q.length);
          const rows = e.shape.length === 2 ? e.shape[0] : 1;
          const cols = q.length / rows;
          for (let r = 0; r < rows; r++) {
            const sv = sc[r];
            for (let c = 0; c < cols; c++) f[r * cols + c] = q[r * cols + c] * sv;
          }
          t[e.name.replace(/\.q$/, "")] = { f32: f, shape: e.shape };
        } else {
          t[e.name] = { f32: new Float32Array(slice), shape: e.shape };
        }
      }
      const m = new PaletteModel(t);
      m.meta = meta; m.cb = cb;
      m.charMap = new Map(vocab.chars.map((c, i) => [c, i + CHAR_BASE]));
      m.wordMap = new Map((vocab.words || []).map((w, i) => [w, i + WORD_BASE]));
      m.nameMap = new Map((vocab.names || []).map((n, i) => [n, i + NAME_BASE]));
      const idx = new Map((vocab.names || []).map((n, i) => [n, i]));
      for (const [v, canon] of Object.entries(vocab.variants || {})) {
        if (idx.has(canon)) m.nameMap.set(v, idx.get(canon) + NAME_BASE);
      }
      m.nameMax = [...m.nameMap.keys()].reduce((a, n) => Math.max(a, n.length), 0);
      m.cbLab = cb.lab.map(a => [a[0], a[1], a[2]]);
      return m;
    }

    tensor(n) { const x = this.t[n]; if (!x) throw new Error("no tensor " + n); return x.f32; }

    _isLatin(c) {
      return (c >= "a" && c <= "z") || (c >= "A" && c <= "Z") || (c >= "0" && c <= "9") || c === "'" || c === "-";
    }

    _knownWord(w) {
      const lw = w.toLowerCase();
      if (this.wordMap.has(lw)) return lw;
      for (const suf of ["s", "es", "ed", "ing", "er", "ly"]) {
        if (lw.length > suf.length + 2 && lw.endsWith(suf)) {
          const stem = lw.slice(0, -suf.length);
          if (this.wordMap.has(stem)) return stem;
          if (this.wordMap.has(stem + "e")) return stem + "e";
        }
      }
      let best = "";
      for (const k of this.wordMap.keys()) {
        if (k.length >= 4 && k.length > best.length && lw.startsWith(k)) best = k;
      }
      return best || null;
    }

    encode(text) {
      // greedy longest match: whole character names first, then latin words, then chars
      const chars = Array.from(text);
      const ids = [BOS];
      let known = true, i = 0, n = 0;
      while (i < chars.length && n < 32) {
        let hitLen = 0, hitId = null;
        for (let ln = Math.min(this.nameMax, chars.length - i); ln > 0; ln--) {
          const cand = chars.slice(i, i + ln).join("");
          if (this.nameMap.has(cand)) { hitLen = ln; hitId = this.nameMap.get(cand); break; }
        }
        if (hitId !== null) { ids.push(hitId); i += hitLen; n++; continue; }
        const c = chars[i];
        if (this._isLatin(c)) {
          let j = i;
          while (j < chars.length && this._isLatin(chars[j])) j++;
          const word = chars.slice(i, j).join("");
          const hit = this._knownWord(word);
          if (hit) ids.push(this.wordMap.get(hit));
          else { ids.push(UNK); known = false; }
          i = j; n++; continue;
        }
        const v = this.charMap.get(c);
        if (v === undefined) known = false;
        ids.push(v === undefined ? UNK : v);
        i++; n++;
      }
      ids.push(EOS);
      this._lastKnown = known;
      return new Int32Array(ids);
    }

    forward(ids, langId) {
      const D = this.meta.d, H = this.meta.heads, dh = D / H;
      const FF = this.t["blocks.0.fc1.weight"].shape[0];
      const T = ids.length;
      const emb = this.tensor("emb.weight"), lem = this.tensor("lang.weight");
      const x = new Float32Array(T * D);
      for (let t = 0; t < T; t++) for (let d = 0; d < D; d++)
        x[t * D + d] = emb[ids[t] * D + d] + lem[langId * D + d];

      for (let l = 0; l < this.meta.layers; l++) {
        const p = "blocks." + l + ".";
        const h = new Float32Array(T * D);
        const kv = new Array(T);
        for (let t = 0; t < T; t++) {
          const row = x.subarray(t * D, t * D + D);
          const y = layerNorm(row, this.tensor(p + "ln1.weight"), this.tensor(p + "ln1.bias"), 1e-6);
          const qkv = linear(y, this.tensor(p + "qkv.weight"), this.tensor(p + "qkv.bias"), 3 * D, D);
          for (let d = 0; d < D; d++) h[t * D + d] = qkv[d];
          kv[t] = { k: qkv.subarray(D, 2 * D), v: qkv.subarray(2 * D, 3 * D) };
        }
        const attn = new Float32Array(T * D);
        for (let head = 0; head < H; head++) {
          const off = head * dh, scale = 1 / Math.sqrt(dh);
          for (let i = 0; i < T; i++) {
            const qi = h.subarray(i * D + off, i * D + off + dh);
            const scores = new Float32Array(T);
            let mx = -1e30;
            for (let j = 0; j < T; j++) {
              const ki = kv[j].k;
              let s = 0;
              for (let d = 0; d < dh; d++) s += qi[d] * ki[off + d];
              s *= scale; scores[j] = s; if (s > mx) mx = s;
            }
            let sum = 0;
            for (let j = 0; j < T; j++) { scores[j] = Math.exp(scores[j] - mx); sum += scores[j]; }
            const oi = i * D + off;
            for (let j = 0; j < T; j++) {
              const w = scores[j] / sum, vi = kv[j].v;
              for (let d = 0; d < dh; d++) attn[oi + d] += w * vi[off + d];
            }
          }
        }
        for (let t = 0; t < T; t++) {
          const a = linear(attn.subarray(t * D, t * D + D), this.tensor(p + "proj.weight"), this.tensor(p + "proj.bias"), D, D);
          for (let d = 0; d < D; d++) x[t * D + d] += a[d];
        }
        for (let t = 0; t < T; t++) {
          const row = x.subarray(t * D, t * D + D);
          const y = layerNorm(row, this.tensor(p + "ln2.weight"), this.tensor(p + "ln2.bias"), 1e-6);
          const h1 = linear(y, this.tensor(p + "fc1.weight"), this.tensor(p + "fc1.bias"), FF, D);
          for (let i = 0; i < h1.length; i++) h1[i] = gelu(h1[i]);
          const h2 = linear(h1, this.tensor(p + "fc2.weight"), this.tensor(p + "fc2.bias"), D, FF);
          for (let d = 0; d < D; d++) x[t * D + d] += h2[d];
        }
      }
      const pooled = new Float32Array(D);
      for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) pooled[d] += x[t * D + d];
      for (let d = 0; d < D; d++) pooled[d] /= T;
      const normed = layerNorm(pooled, this.tensor("ln.weight"), this.tensor("ln.bias"), 1e-6);
      const N = this.meta.ncolor;
      const logits = linear(normed, this.tensor("head.weight"), this.tensor("head.bias"), 6 * N, D);
      const logitsImg = this.t["head_img.weight"]
        ? linear(normed, this.tensor("head_img.weight"), this.tensor("head_img.bias"), 6 * N, D)
        : logits;
      const conf = sigmoid(linear(normed, this.tensor("conf.weight"), this.tensor("conf.bias"), 1, D)[0]);
      return { logits, logitsImg, conf };
    }

    labDist(a, b) {
      const dl = a[0] - b[0], da = a[1] - b[1], db = a[2] - b[2];
      return Math.sqrt(dl * dl + da * da + db * db);
    }

    palette(text, opts) {
      opts = opts || {};
      const lang = LANG[opts.lang] || 0;
      const mix = Math.max(0, Math.min(1, opts.mix === undefined ? 0.0 : opts.mix));
      const S = 6, N = this.meta.ncolor, K = opts.k || 6;
      const ids = this.encode(text);
      const { logits, logitsImg, conf } = this.forward(ids, lang);

      /* blend the two heads at probability level */
      const P = [];
      for (let s = 0; s < S; s++) {
        const off = s * N;
        let m1 = -1e30, m2 = -1e30;
        for (let i = 0; i < N; i++) { m1 = Math.max(m1, logits[off + i]); m2 = Math.max(m2, logitsImg[off + i]); }
        let z1 = 0, z2 = 0;
        const a = new Float32Array(N), b = new Float32Array(N);
        for (let i = 0; i < N; i++) {
          a[i] = Math.exp(logits[off + i] - m1); z1 += a[i];
          b[i] = Math.exp(logitsImg[off + i] - m2); z2 += b[i];
        }
        const p = new Float32Array(N);
        for (let i = 0; i < N; i++) p[i] = (1 - mix) * a[i] / z1 + mix * b[i] / z2;
        P.push(p);
      }

      /* rank fusion: a colour scores by how highly EVERY slot ranks it,
         not by the single best slot (max_s let one slot monopolise the palette) */
      const SLOT_W = [1.0, 0.85, 0.7, 0.6, 0.5, 0.45];
      const dom = new Float32Array(N), fus = new Float32Array(N), prob = new Float32Array(N);
      for (let s = 0; s < S; s++) {
        const idx = Array.from({ length: N }, (_, i) => i).sort((x, y) => P[s][y] - P[s][x]);
        for (let r = 0; r < N; r++) {
          const c = idx[r];
          fus[c] += SLOT_W[s] * (1 - r / N);
          const v = SLOT_W[s] * P[s][c];
          if (v > dom[c]) dom[c] = v;
          if (P[s][c] > prob[c]) prob[c] = P[s][c];
        }
      }
      for (let i = 0; i < N; i++) fus[i] /= S;
      let domMax = 0;
      for (let i = 0; i < N; i++) if (dom[i] > domMax) domMax = dom[i];
      const score = new Float32Array(N);
      for (let i = 0; i < N; i++) score[i] = 0.6 * (dom[i] / (domMax || 1)) + 0.4 * fus[i];

      /* hard diversity floor first, score order second — MMR only breaks ties.
         Relax the floor only if 16 cannot fill the palette. */
      const byDom = Array.from({ length: N }, (_, i) => i).sort((a, b) => dom[b] - dom[a]);
      const byFus = Array.from({ length: N }, (_, i) => i).sort((a, b) => score[b] - score[a]);
      const picked = [byDom[0]];                    /* dominant decided by probability */
      const deMin = opts.de === undefined ? 16 : opts.de;
      for (const floor of [deMin, deMin * 0.75, deMin * 0.5, 0]) {
        for (const c of byFus) {
          if (picked.length >= K) break;
          if (picked.indexOf(c) >= 0) continue;
          let ok = true;
          for (const q of picked) if (this.labDist(this.cbLab[c], this.cbLab[q]) < floor) { ok = false; break; }
          if (ok) picked.push(c);
        }
        if (picked.length >= K) break;
      }

      const out = picked.map(i => ({
        hex: this.cb.hex[i],
        score: +score[i].toFixed(4),
        p: +prob[i].toFixed(4),
        lab: this.cb.lab[i].map(v => +v.toFixed(1)),
      }));

      /* adaptive k: the tail is often filler. Cut where the probability drops
         below 2% of the dominant, but never below 4 colours. */
      const p0 = out[0].p;
      let cut = out.length;
      for (let i = 4; i < out.length; i++) if (out[i].p < 0.02 * p0) { cut = i; break; }
      const strong = out.slice(0, cut);
      const extra = out.slice(cut).map(o => Object.assign({}, o, { extended: true }));

      /* the palette is usually a single-hue scale plus one contrast colour:
         expose it that way so a UI does not pretend it is six equal swatches */
      const byL = strong.slice().sort((a, b) => a.lab[0] - b.lab[0]);
      const domLab = strong[0].lab;
      const hueOf = o => (Math.atan2(o.lab[2], o.lab[1]) * 180 / Math.PI + 360) % 360;
      let contrast = null, far = -1;
      for (const o of strong) {
        if (Math.hypot(o.lab[1], o.lab[2]) < 12) continue;
        let d = Math.abs(hueOf(o) - hueOf({ lab: domLab })) % 360;
        d = d > 180 ? 360 - d : d;
        if (d > far) { far = d; contrast = o; }
      }
      if (far < 40) contrast = null;               /* not a real accent, just a darker step */
      const scale = byL.filter(o => o !== contrast);
      return {
        confidence: +conf.toFixed(3), mix: +mix.toFixed(2), colors: strong,
        extra, k: strong.length,
        scale, contrast, hueSpread: Math.round(far),
        known: this._lastKnown !== false,
      };
    }
  }

  global.PaletteNet = { PaletteModel, load: PaletteModel.load };
})(typeof window !== "undefined" ? window : globalThis);
