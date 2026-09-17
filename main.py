"""Trace the Ace inference: 9B KT (single pass) + KC-mastery/TIM features.

Pipeline per session:
  1. TIM: 27 provider-invariant transcript features (numpy port of
     extract_hardened.py, train-median imputation for invalid timing).
  2. Pairs: tutor->student exchanges; assessable = digits OR >=8 words.
  3. MiniLM embeds each assessable pair -> nearest of 2,000 cluster
     centers -> that cluster's CCSS-standard KCs (labeled offline).
  4. Qwen3.5-2B judge (distilled from a teacher model) -> P(true/false/na)
     per pair.
  5. LO -> KCs: MiniLM cosine vs 229 CCSS standard descriptions (top-4,
     rel >= 0.85) -- zero train-time dependence on curriculum phrasing.
  6. Mastery features (relevant-KC weighted correctness w/ hierarchical
     backoff + objective-coverage count) + TIM -> logistic mapper -> z_feat.
  7. Qwen3.5-9B KT (BCE-trained on real quiz outcomes, one LoRA adapter
     over the base): transcript tail (3,400 tok) + objective -> P(True);
     z_kt = logit / T + b with T=1.6, b=+0.148 (calibrated on the
     unseen-objective cell -- the KT is over-confident there).
     The adapter loop / time guard remain but ship with a single adapter.
  8. blend z = 0.6*z_kt + 0.4*z_feat, lam=1.3 (unseen-objective fit).

Failure ladder: any per-response model failure -> feature-only; feature
failure -> prior. A global time guard flips remaining work to the
feature-only path if the model stages threaten the runtime limit.

The vendored `vendor/` directory (transformers 5.x + deps) is prepended to
sys.path because the runtime image pins transformers 4.57, which predates
the qwen3_5 architecture. Binary deps (tokenizers) come from the image.
"""

import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Pure-python vendored libs (transformers 5.x, huggingface_hub) ship as one
# vendor_py.zip to keep the submission's member count low (the runtime logs
# every zip member and the platform caps logs at 500 lines). transformers
# scans its models/ dir with os.scandir, so zipimport won't do -- extract to
# a temp dir at startup. Binary packages (safetensors, hf_xet) stay as real
# directories in vendor/.
import tempfile
import zipfile as _zf

_vzip = ROOT / "vendor" / "vendor_py.zip"
if _vzip.exists():
    _vdir = Path(tempfile.mkdtemp(prefix="vendor_py_"))
    with _zf.ZipFile(_vzip) as _z:
        _z.extractall(_vdir)
    sys.path.insert(0, str(_vdir))
sys.path.insert(0, str(ROOT / "vendor"))

import numpy as np                                          # noqa: E402

DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else Path("data")
ASSETS = ROOT / "assets"
MODELS = ROOT / "models"
SUBMISSION_PATH = Path("submission.csv")

T0 = time.time()
TIME_LIMIT = 6 * 3600
MODEL_DEADLINE = T0 + TIME_LIMIT - 45 * 60   # leave 45 min of slack

C = json.load(open(ASSETS / "constants.json"))
WORD_RE = re.compile(r"[a-z']+")
PAIR_WORD_RE = re.compile(r"[a-z0-9']+", re.I)
UNCERTAIN = ("i don't know", "i dont know", "not sure", "confused",
             "don't understand", "dont understand", "don't get", "dont get")
AGREE = ("yes", "yeah", "yep", "ok", "okay", "mhm", "mm")


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def logit(p):
    p = min(max(p, 1e-7), 1 - 1e-7)
    return float(np.log(p / (1 - p)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- transcripts
def parse_ts(s):
    try:
        p = [int(x) for x in str(s).strip().split(":")]
    except ValueError:
        return None
    return sum(v * m for v, m in zip(p[::-1], (1, 60, 3600))) \
        if len(p) in (2, 3) else None


def read_transcript(session_id):
    path = DATA_DIR / "test_transcripts" / f"{session_id}.csv"
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [(str(x.get("role", "")).strip().lower(),
                 (x.get("content") or ""),
                 parse_ts(x.get("timestamp")))
                for x in csv.DictReader(f)]


# ------------------------------------------------- TIM features (verbatim port)
def toks(s):
    return WORD_RE.findall(s.lower())


def tim_features(turns):
    n = len(turns)
    stu = [(c, t) for r, c, t in turns if r == "student"]
    tut = [(c, t) for r, c, t in turns if r != "student"]
    ns, nt = len(stu), max(len(tut), 1)
    s_chars = sum(len(c) for c, _ in stu)
    t_chars = sum(len(c) for c, _ in tut)
    s_words = [toks(c) for c, _ in stu]
    wlens = [len(w) for w in s_words]

    third = max(n // 3, 1)
    early = sum(len(c) for r, c, _ in turns[:third] if r == "student")
    late = sum(len(c) for r, c, _ in turns[-third:] if r == "student")
    e_tot = sum(len(c) for _, c, _ in turns[:third]) or 1
    l_tot = sum(len(c) for _, c, _ in turns[-third:]) or 1

    alt = sum(1 for i in range(1, n)
              if (turns[i][0] == "student") != (turns[i - 1][0] == "student"))
    runs, cur = [], 0
    for r, _, _ in turns:
        if r != "student":
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)

    low = " ".join(c for c, _ in stu).lower()
    unc = sum(low.count(k) for k in UNCERTAIN)
    agree = sum(1 for w in s_words
                if w and all(x in AGREE for x in w[:2]) and len(w) <= 2)
    digit_turns = sum(1 for c, _ in stu if any(ch.isdigit() for ch in c))

    row = [
        np.log1p(n),
        ns / max(n, 1),
        s_chars / max(s_chars + t_chars, 1),
        np.log((s_chars / max(ns, 1) + 1) / (t_chars / nt + 1)),
        np.log1p(np.median(wlens) if wlens else 0),
        (sum(1 for w in wlens if w < 4) / max(ns, 1)),
        (sum(1 for w in wlens if w == 1) / max(ns, 1)),
        sum(c.count("?") for c, _ in tut) / nt,
        sum(c.count("?") for c, _ in stu) / max(ns, 1),
        alt / max(n - 1, 1),
        float(np.mean(runs)) if runs else 0.0,
        digit_turns / max(ns, 1),
        late / l_tot - early / e_tot,
        np.log1p(s_chars),
        unc / max(ns, 1),
        agree / max(ns, 1),
        np.log((ns + 1) / (nt + 1)),
        sum(ch.isdigit() for c, _ in stu for ch in c) / max(sum(wlens), 1),
    ]

    ts = [t for _, _, t in turns if t is not None]
    valid = (len(ts) >= 0.8 * n and len(set(ts)) > 3
             and n > 3 and (max(ts) - min(ts)) > 0)
    if valid:
        gaps = np.maximum(np.diff([t for _, _, t in turns
                                   if t is not None]), 0).astype(float)
        med = float(np.median(gaps))
        p90 = float(np.percentile(gaps, 90))
        pre_s = [max(turns[i][2] - turns[i - 1][2], 0)
                 for i in range(1, n)
                 if turns[i][0] == "student"
                 and turns[i][2] is not None and turns[i - 1][2] is not None]
        pre_t = [max(turns[i][2] - turns[i - 1][2], 0)
                 for i in range(1, n)
                 if turns[i][0] != "student"
                 and turns[i][2] is not None and turns[i - 1][2] is not None]
        dur = max(ts) - min(ts)
        half = len(gaps) // 2 or 1
        row += [
            np.log1p(dur),
            np.log1p(med),
            np.log1p(p90),
            np.log((p90 + 1) / (med + 1)),
            np.log((np.median(pre_s) + 1) / (np.median(pre_t) + 1)
                   ) if pre_s and pre_t else 0.0,
            float(np.mean(gaps > 3 * max(med, 1))),
            np.log((np.mean(gaps[half:]) + 1) / (np.mean(gaps[:half]) + 1)),
            np.log1p(60.0 * n / max(dur, 1)),
            1.0,
        ]
    else:
        row += [np.nan] * 8 + [0.0]
    return np.array(row, dtype=np.float64)


# ------------------------------------------------------------------- pairs
def build_pairs(raw):
    turns = []
    for role, text, ts in raw:
        text = text.strip()
        if role == "background" or not text:
            continue
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + " " + text)
        else:
            turns.append((role, text))
    pairs, pending = [], None
    for role, text in turns:
        if role == "student":
            pairs.append((pending or "", text))
            pending = None
        else:
            pending = text
    return pairs                       # [(tutor_text, student_text), ...]


def assessable(student_text):
    if any(ch.isdigit() for ch in student_text):
        return True
    return len(PAIR_WORD_RE.findall(student_text)) >= C["assessable_min_words"]


# -------------------------------------------------------------- KC machinery
def kc_levels(code):
    p = code.split(".")
    return code, ".".join(p[:3]), p[1]


def mastery_row(ev, target_codes):
    """ev: [(levels3, corr, weight)]; 7 features as trained (mas7)."""
    tsets = [set(), set(), set()]
    for code in target_codes:
        for L, part in enumerate(kc_levels(code)):
            tsets[L].add(part)
    wa = sum(w for _, _, w in ev)
    overall = (sum(c * w for _, c, w in ev) / wa) if wa > 1e-6 else 0.5
    score, lev, wr = overall, 3, 0.0
    for L in range(3):
        if not tsets[L]:
            continue
        rel = [(c, w) for lv, c, w in ev if lv[L] & tsets[L]]
        wrel = sum(w for _, w in rel)
        if wrel > 1e-6:
            score = sum(c * w for c, w in rel) / wrel
            lev, wr = L, wrel
            break
    # 7th feature: coverage -- number of judged pairs whose KC matches the
    # objective's KCs at ANY level (log1p); trained as mas7 in mapper.npz
    n_rel = sum(1 for lv, _, _ in ev
                if any(lv[L] & tsets[L] for L in range(3) if tsets[L]))
    return np.array([score, overall, lev, np.log1p(wr), np.log1p(wa),
                     score - overall, np.log1p(n_rel)], dtype=np.float64)


# ------------------------------------------------------------------- models
def load_tok_model(path, causal, dtype):
    from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer)
    import torch
    tok = AutoTokenizer.from_pretrained(str(path), padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cls = AutoModelForCausalLM if causal else AutoModel
    model = cls.from_pretrained(str(path), dtype=dtype,
                                attn_implementation="sdpa").cuda().eval()
    return tok, model


def minilm_embed(texts, tok, model, bs=256, maxtok=96):
    import torch
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=maxtok).to("cuda")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            e = (h * m).sum(1) / m.sum(1)
            e = torch.nn.functional.normalize(e, dim=-1)
        out.append(e.float().cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 384), np.float32)


JUDGE_SYSTEM = (
    "You are an experienced math teacher reviewing a transcribed SPOKEN "
    "dialogue between a tutor and a student in an online math tutoring "
    "session. Transcripts contain filler words and [unclear] where audio was "
    "lost, and speaker attribution is occasionally imperfect. Judge whether "
    "the student correctly answered the tutor's question in the FINAL "
    "exchange. Respond with exactly one word: True if the student's answer "
    "to an assessing question is correct; False if the tutor asked an "
    "assessing question with a correct answer and the student answered "
    "incorrectly or not at all; NA if the tutor's turn does not assess "
    "mathematical knowledge, has no right or wrong answer, or the student "
    "turn is too garbled to judge.")

KT_SYSTEM = (
    "You are an expert mathematics tutor evaluator. You will read the end of "
    "a tutoring session transcript, then judge whether the student will "
    "answer a quiz question on the given learning objective correctly, "
    "immediately after the session. Respond with exactly one word: True if "
    "the student will answer correctly, False otherwise.")


def chat(tok, system, user):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    try:
        p = tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    if p.rstrip().endswith("<think>"):
        p = p.rstrip()[: -len("<think>")] + "<think>\n\n</think>\n\n"
    return p


def judge_prompt(tok, ctx_pairs, pair):
    tcut, scut = C["judge_tcut"], C["judge_scut"]
    lines = []
    for t, s in ctx_pairs:
        if t:
            lines.append(f"Tutor: {t[:tcut]}")
        lines.append(f"Student: {s[:scut]}")
    lines.append("--- FINAL EXCHANGE ---")
    if pair[0]:
        lines.append(f"Tutor: {pair[0][:tcut]}")
    lines.append(f"Student: {pair[1][:scut]}")
    user = ("\n".join(lines) +
            "\n\nDid the student answer the tutor's question in the final "
            "exchange correctly? Answer True, False, or NA.")
    return chat(tok, JUDGE_SYSTEM, user)


def token_logits(model, enc, ids):
    import torch
    h = model.model(input_ids=enc["input_ids"].cuda(),
                    attention_mask=enc["attention_mask"].cuda()
                    ).last_hidden_state
    last = (enc["attention_mask"].sum(1) - 1).cuda()
    hh = h[torch.arange(len(last), device=h.device), last]
    return model.lm_head(hh)[:, ids].float()


def batched_forced_choice(rows, tok, model, ids, micro, max_len, tag):
    """rows: [(key, prompt)] -> {key: softmax probs over ids}."""
    import torch
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][1]))
    out = {}
    with torch.no_grad():
        for i in range(0, len(order), micro):
            chunk = [rows[j] for j in order[i:i + micro]]
            enc = tok([p for _, p in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=max_len,
                      add_special_tokens=False)
            P = torch.softmax(token_logits(model, enc, ids), -1).cpu().numpy()
            for (k, _), p in zip(chunk, P):
                out[k] = p
            if (i // micro) % 200 == 0:
                log(f"  {tag} {i}/{len(order)}")
            if time.time() > MODEL_DEADLINE:
                log(f"  {tag}: DEADLINE hit at {i}/{len(order)}; "
                    "remaining rows fall back")
                break
    return out


# --------------------------------------------------------------------- main
def main():
    feats = list(csv.DictReader(open(DATA_DIR / "test_features.csv")))
    sub = list(csv.DictReader(open(DATA_DIR / "submission_format.csv")))
    sessions = sorted({r["session_id"] for r in feats})
    log(f"{len(feats)} responses, {len(sessions)} sessions")

    m = np.load(ASSETS / "mapper.npz")
    centers = np.load(ASSETS / "centers.npy")          # (2000, 384)
    clus_kcs = {int(k): v for k, v in
                json.load(open(ASSETS / "cluster_kcs.json")).items()}
    ccss = np.load(ASSETS / "ccss.npz", allow_pickle=True)
    ccss_codes = [str(x) for x in ccss["codes"]]
    ccss_emb = ccss["emb"]
    prior = C["prior"]
    z0 = logit(prior)

    # ---- pass 1: transcripts -> TIM features + pairs (CPU) --------------
    tim = {}
    pairs_by = {}
    for s in sessions:
        try:
            raw = read_transcript(s)
            if not raw:
                raw = [("tutor", "", None)]
            tim[s] = tim_features(raw)
            pairs_by[s] = build_pairs(raw)
        except Exception as e:                                  # noqa: BLE001
            log(f"  transcript fail {s}: {e}")
            tim[s] = None
            pairs_by[s] = []
    log("transcripts parsed")

    # ---- pass 2: MiniLM -> cluster assign + LO->KC ----------------------
    lo_codes = {}
    clus_of = defaultdict(dict)
    try:
        tok_e, mod_e = load_tok_model(MODELS / "minilm", causal=False,
                                      dtype=None)
        pair_keys, pair_texts = [], []
        for s in sessions:
            for i, (t, st) in enumerate(pairs_by[s]):
                if assessable(st):
                    pair_keys.append((s, i))
                    pair_texts.append(
                        f"T: {t[:C['pair_tcut']]} S: {st[:C['pair_scut']]}")
        E = minilm_embed(pair_texts, tok_e, mod_e)
        for k in range(0, len(E), 100_000):
            d = E[k:k + 100_000] @ centers.T
            lab = d.argmax(1)
            for (key, c) in zip(pair_keys[k:k + 100_000], lab):
                clus_of[key[0]][key[1]] = int(c)
        log(f"{len(pair_keys)} assessable pairs clustered")

        los = {}
        for r in feats:
            los.setdefault(r["learning_objective_id"],
                           r["learning_objective"].strip())
        lo_ids = sorted(los)
        EL = minilm_embed([los[i] for i in lo_ids], tok_e, mod_e)
        S = EL @ ccss_emb.T
        for i, oid in enumerate(lo_ids):
            srow = S[i]
            top = np.argsort(-srow)[:C["lo_topk"]]
            best = srow[top[0]]
            lo_codes[oid] = [ccss_codes[j] for j in top
                             if srow[j] >= C["lo_rel"] * best]
        log(f"{len(lo_ids)} objectives mapped to KCs")
        del mod_e
        import torch
        torch.cuda.empty_cache()
    except Exception as e:                                      # noqa: BLE001
        log(f"MiniLM stage failed ({e}); mastery falls back to defaults")

    # ---- pass 3: 2B judge on assessable pairs ---------------------------
    pair_probs = {}
    try:
        tok_j, mod_j = load_tok_model(MODELS / "judge2b", causal=True,
                                      dtype="bfloat16")
        jt = [tok_j.encode(w, add_special_tokens=False)[0]
              for w in ("True", "False", "NA")]
        rows = []
        for s in sessions:
            pl = pairs_by[s]
            for i in sorted(clus_of.get(s, {})):
                ctx = pl[max(i - C["judge_ctx"], 0):i]
                rows.append(((s, i), judge_prompt(tok_j, ctx, pl[i])))
        pair_probs = batched_forced_choice(
            rows, tok_j, mod_j, jt, micro=64, max_len=1024, tag="judge")
        log(f"{len(pair_probs)} pairs judged")
        del mod_j
        import torch
        torch.cuda.empty_cache()
    except Exception as e:                                      # noqa: BLE001
        log(f"judge stage failed ({e}); mastery falls back to defaults")

    # ---- pass 4: 9B KT seed ensemble (one base + LoRA adapter passes) ---
    # kt_z: response_id -> [raw logit per completed pass]. Passes run in a
    # fixed seed order; the time guard decides how many run, and the
    # calibration table in constants.json is indexed by that count.
    kt_z = {}
    try:
        tok_k, mod_k = load_tok_model(MODELS / "kt9b_base", causal=True,
                                      dtype="bfloat16")
        from peft import PeftModel
        adir = Path(tempfile.mkdtemp(prefix="kt_adapters_"))
        with _zf.ZipFile(MODELS / "kt_adapters.zip") as _z:
            _z.extractall(adir)
        id_t = tok_k.encode("True", add_special_tokens=False)[0]
        id_f = tok_k.encode("False", add_special_tokens=False)[0]
        tails = {}
        for s in sessions:
            try:
                body = "\n".join(f"{r}: {c}" for r, c, _ in
                                 read_transcript(s))
            except Exception:                                   # noqa: BLE001
                body = ""
            ids = tok_k.encode(body, add_special_tokens=False)
            if len(ids) > C["kt_max_tr"]:
                body = tok_k.decode(ids[-C["kt_max_tr"]:])
            tails[s] = body
        rows = []
        for r in feats:
            user = (f"Transcript (end of session):\n{tails[r['session_id']]}"
                    f"\n\nLearning objective of the quiz question:\n"
                    f"{r['learning_objective'].strip()}\n\n"
                    "Will the student answer the quiz question correctly? "
                    "Answer True or False.")
            rows.append((r["response_id"], chat(tok_k, KT_SYSTEM, user)))
        for ai, name in enumerate(C["kt_adapters"]):
            apath = str(adir / name)
            if ai == 0:
                mod_k = PeftModel.from_pretrained(mod_k, apath,
                                                  adapter_name=name)
            else:
                mod_k.load_adapter(apath, adapter_name=name)
                mod_k.set_adapter(name)
            mod_k.eval()
            t_pass = time.time()
            # forward through the LoRA-injected causal LM inside the peft
            # wrapper: token_logits needs .model (backbone) / .lm_head
            got = batched_forced_choice(
                rows, tok_k, mod_k.model, [id_t, id_f], micro=16,
                max_len=C["kt_max_len"], tag=f"kt-{name}")
            dur = time.time() - t_pass
            if len(got) < len(rows) and ai > 0:
                # deadline tripped mid-pass: a partial later pass would give
                # rows unequal pass counts -> discard it, keep prior passes
                log(f"  kt-{name} partial ({len(got)}/{len(rows)}), "
                    "discarded")
                break
            for k, v in got.items():
                kt_z.setdefault(k, []).append(logit(float(v[0])))
            log(f"kt pass {ai + 1} ({name}): {len(got)} rows "
                f"in {dur / 60:.1f} min")
            if ai + 1 == len(C["kt_adapters"]):
                break
            if time.time() + 1.15 * dur > MODEL_DEADLINE:
                log(f"  no time for pass {ai + 2} "
                    f"(~{dur / 60:.0f} min); stopping ensemble here")
                break
        del mod_k
    except Exception as e:                                      # noqa: BLE001
        log(f"KT stage failed ({e}); shipping feature-only predictions")

    # ---- pass 5: assemble -----------------------------------------------
    ev_by = {}
    for s in sessions:
        ev = []
        for i, c in clus_of.get(s, {}).items():
            p = pair_probs.get((s, i))
            if p is None or c not in clus_kcs:
                continue
            lv = [set(), set(), set()]
            for code in clus_kcs[c]:
                for L, part in enumerate(kc_levels(code)):
                    lv[L].add(part)
            corr = p[0] / max(p[0] + p[1], 1e-6)
            ev.append((lv, float(corr), float(1.0 - p[2])))
        ev_by[s] = ev

    preds = {}
    n_full = n_feat = n_prior = 0
    for r in feats:
        rid, s = r["response_id"], r["session_id"]
        try:
            F = tim[s]
            if F is None:
                raise ValueError("no transcript features")
            Fi = F.copy()
            bad = np.isnan(Fi)
            Fi[bad] = m["tim_median"][bad]
            M6 = mastery_row(ev_by.get(s, []),
                             lo_codes.get(r["learning_objective_id"], []))
            x = np.concatenate([Fi, M6])
            xs = (x - m["scaler_mean"]) / m["scaler_std"]
            z_feat = float(xs @ m["coef"] + m["intercept"][0])
            zs_r = kt_z.get(rid)
            if zs_r:
                Tk, bk = C["kt_cal"][str(len(zs_r))]
                z_kt = (sum(zs_r) / len(zs_r)) / Tk + bk
                z = C["w_kt"] * z_kt + (1 - C["w_kt"]) * z_feat
                n_full += 1
            else:
                z = z_feat
                n_feat += 1
            p = sigmoid(z0 + C["lam"] * (z - z0))
        except Exception:                                       # noqa: BLE001
            p = prior
            n_prior += 1
        preds[rid] = min(max(float(p), 0.01), 0.99)

    with open(SUBMISSION_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["response_id", "probability"])
        for r in sub:
            w.writerow([r["response_id"],
                        f"{preds.get(r['response_id'], prior):.6f}"])
    log(f"wrote {len(sub)} rows: {n_full} blended, {n_feat} feature-only, "
        f"{n_prior} prior")


if __name__ == "__main__":
    main()
