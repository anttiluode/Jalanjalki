#!/usr/bin/env python3
"""
Jalanjalki G8b — same axis, new coupling?  Or a recode at the answer boundary?

Run 3 (G8) fitted a SEPARATE post-training direction at each of three positions
(user_end, turn_end, assistant_start) and found:
    d'            0.92 -> 0.79 -> 1.36
    rho(policy)   0.15 -> 0.27 -> 0.63
That cannot tell two stories apart:

  A. PHASE GATING  ("same information, new coupling")
     The direction that carries post-training's recognition at user_end is the
     same direction at assistant_start.  What changes at the boundary is how
     strongly that direction is wired to the act (Hasselmo-style encoding vs
     retrieval phase; AIS-style gate).
  B. RECODE
     At the boundary the network rewrites the recognition into a different,
     action-oriented axis.  The user_end direction stops carrying the signal.

G8b tests it.  For every layer in the band and every pair of positions (p, q):
  * fit v_p on the MAIN prompts at position p (instruct-minus-base,
    sensitive-minus-neutral; exactly run 3's direction)
  * evaluate on the 24 HELD-OUT matched pairs at position q:
      d'(p->q)     post-training's loaded-minus-benign separation
      rho(p->q)    Spearman of the projection with the unsteered policy score
  * cos(v_p, v_q), with a label-shuffle null (shuffled sensitive/neutral labels)
  * decomposition at assistant_start: split v_asst into its component along
    v_user and the orthogonal remainder; which part carries rho(policy)?

PRE-REGISTERED VERDICT (band means; written before any real-model run):
  SAME AXIS (supports A) if all three:
     cos(v_user, v_asst) above the 95th percentile of the shuffle null
     d'(user->asst) >= 0.5 * d'(asst->asst)
     rho(user->asst) >= rho(user->user) + 0.2
  RECODE (supports B) if:
     d'(user->asst) < 0.3 * d'(asst->asst)  OR  cos not above null,
     AND the orthogonal remainder carries more rho than the v_user component
  otherwise MIXED.

Cheap: two forward passes per prompt per model plus policy scores; no generation.

Run:  python jalanjalki_g8b.py            (Qwen3-0.6B pair, same as run 3)
      python jalanjalki_g8b.py --smoke    (tiny random models, no downloads)
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import torch

import jalanjalki as JJ
import jalanjalki_run3 as R3

POS = R3.POSITIONS            # user_end, turn_end, assistant_start
SHORT = {"user_end": "user", "turn_end": "turn", "assistant_start": "asst"}


def unit(v):
    return v / (v.norm() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B")
    ap.add_argument("--band", default=None, help="e.g. 15,20 (default: 55-75%% depth, as run 3)")
    ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default="results/g8b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    # ------------------------------------------------------------ same prompt layout as run 3
    main_sets = ["neutral", "request", "sensitive"]
    contents = [p for s in main_sets for p in JJ.PROMPTS[s]]
    set_of = [s for s in main_sets for _ in JJ.PROMPTS[s]]
    for _, b, l in JJ.PAIRS:
        contents += [b, l]
        set_of += ["pair_benign", "pair_loaded"]
    idx = {s: [i for i, x in enumerate(set_of) if x == s] for s in set(set_of)}
    PB, PL = idx["pair_benign"], idx["pair_loaded"]
    held = PB + PL
    SEN, NEU = idx["sensitive"], idx["neutral"]

    if a.smoke:
        tok = JJ.ToyTok(contents + R3.REFUSE + R3.CAUTION + R3.COMPLY)
        models = dict(zip(["base", "instruct"], JJ.smoke_models(tok, device)))
    else:
        from transformers import AutoTokenizer
        names = {"base": f"Qwen/Qwen3-{a.size}-Base", "instruct": f"Qwen/Qwen3-{a.size}"}
        tok = AutoTokenizer.from_pretrained(names["instruct"])

    # positions sanity: if turn_end collapses onto another position, say so
    _, _, p0 = R3.tpl_and_positions(tok, contents[0])
    print("positions for first prompt:", p0)

    S = {}
    for tag in ["base", "instruct"]:
        print(f"=== {tag}")
        model = models[tag] if a.smoke else JJ.load_model(names[tag], dtype, device)
        st = R3.Steer(model)
        S[tag] = R3.states_at(st, tok, contents, device)
        if tag == "base":
            del st, model
            if device == "cuda":
                torch.cuda.empty_cache()

    # unsteered policy score of the instruct model on the held-out pairs (run 3's G7b)
    st.set()
    pol0 = np.array([R3.policy_score(st, tok, contents[i], device)["policy"] for i in held])

    L = S["base"]["assistant_start"].shape[1]
    band = list(range(round(0.55 * (L - 1)), round(0.75 * (L - 1)) + 1))
    if a.band:
        lo, hi = map(int, a.band.split(","))
        band = list(range(lo, hi + 1))
    print(f"layers={L}  band=L{band[0]}-L{band[-1]}")

    def direction(pos, l, A_, B_):
        I, B = S["instruct"][pos][:, l], S["base"][pos][:, l]
        return (I[A_].mean(0) - I[B_].mean(0)) - (B[A_].mean(0) - B[B_].mean(0))

    def evaluate(vh, pos, l):
        """d' of post-training's per-pair loaded-minus-benign, and rho with policy."""
        pI = S["instruct"][pos][:, l] @ vh
        pB = S["base"][pos][:, l] @ vh
        dd = (pI[PL] - pI[PB]) - (pB[PL] - pB[PB])
        dprime = float(dd.mean() / (dd.std() + 1e-12))
        rho = R3.spearman((pI - pB)[held].numpy(), pol0)
        return dprime, rho

    rng = np.random.default_rng(a.seed)
    pool = SEN + NEU
    nS = len(SEN)

    per_layer = []
    for l in band:
        V = {p: unit(direction(p, l, SEN, NEU)) for p in POS}
        row = {"layer": l, "dprime": {}, "rho": {}, "cos": {}}
        for p in POS:
            for q in POS:
                d_, r_ = evaluate(V[p], q, l)
                row["dprime"][f"{SHORT[p]}->{SHORT[q]}"] = d_
                row["rho"][f"{SHORT[p]}->{SHORT[q]}"] = r_
        for i, p in enumerate(POS):
            for q in POS[i + 1:]:
                row["cos"][f"{SHORT[p]}~{SHORT[q]}"] = float(V[p] @ V[q])

        # label-shuffle null for cos(v_user, v_asst)
        null = []
        for _ in range(a.shuffles):
            perm = rng.permutation(pool)
            A_, B_ = list(perm[:nS]), list(perm[nS:])
            vu = unit(direction("user_end", l, A_, B_))
            va = unit(direction("assistant_start", l, A_, B_))
            null.append(float(vu @ va))
        row["cos_null_95"] = float(np.percentile(null, 95))
        row["cos_null_mean"] = float(np.mean(null))

        # decomposition at assistant_start: v_asst = parallel(v_user) + orthogonal
        vu, va = V["user_end"], V["assistant_start"]
        par = (va @ vu) * vu
        orth = va - par
        row["rho_asst_along_vuser"] = evaluate(vu, "assistant_start", l)[1]
        row["rho_asst_orthogonal"] = evaluate(unit(orth), "assistant_start", l)[1] if orth.norm() > 1e-6 else 0.0
        row["dprime_asst_orthogonal"] = evaluate(unit(orth), "assistant_start", l)[0] if orth.norm() > 1e-6 else 0.0
        per_layer.append(row)

    def bmean(key, sub=None):
        vals = [r[key][sub] if sub else r[key] for r in per_layer]
        return float(np.mean(vals))

    keys = [f"{SHORT[p]}->{SHORT[q]}" for p in POS for q in POS]
    T_d = {k: bmean("dprime", k) for k in keys}
    T_r = {k: bmean("rho", k) for k in keys}
    cos = {k: bmean("cos", k) for k in per_layer[0]["cos"]}
    null95 = bmean("cos_null_95")
    rho_par = bmean("rho_asst_along_vuser")
    rho_orth = bmean("rho_asst_orthogonal")

    print("\n[G8b] transfer matrix, band mean  (row = position the direction was fitted at,"
          " column = position it is tested at)")
    for title, T in [("d' (post-training loaded-minus-benign)", T_d), ("rho with policy score", T_r)]:
        print(f"\n  {title}")
        print("            " + "".join(f"{'@' + SHORT[q]:>9}" for q in POS))
        for p in POS:
            print(f"  v_{SHORT[p]:<8}" + "".join(f"{T[f'{SHORT[p]}->{SHORT[q]}']:+9.2f}" for q in POS))
    print("\n  cosines between fitted directions (band mean):",
          ", ".join(f"{k} {v:+.2f}" for k, v in cos.items()),
          f"| shuffle null 95th pct for user~asst {null95:+.2f}")
    print(f"  at assistant_start, rho carried by v_user component {rho_par:+.2f}, "
          f"by the orthogonal remainder {rho_orth:+.2f}")

    # ------------------------------------------------------------ pre-registered verdict
    c_ua = cos["user~asst"]
    same_axis = (c_ua > null95 and T_d["user->asst"] >= 0.5 * T_d["asst->asst"]
                 and T_r["user->asst"] >= T_r["user->user"] + 0.2)
    recode = ((T_d["user->asst"] < 0.3 * T_d["asst->asst"] or c_ua <= null95)
              and abs(rho_orth) > abs(rho_par))
    verdict = "SAME AXIS (phase gating supported)" if same_axis else \
              "RECODE (phase-gating analogy fails)" if recode else "MIXED"
    print(f"\n  VERDICT: {verdict}")
    print(f"    cos(user,asst) {c_ua:+.2f} vs null95 {null95:+.2f}; "
          f"d' user->asst {T_d['user->asst']:+.2f} vs asst->asst {T_d['asst->asst']:+.2f}; "
          f"rho user->asst {T_r['user->asst']:+.2f} vs user->user {T_r['user->user']:+.2f}")

    rep = {"band": [band[0], band[-1]], "n_layers": L, "transfer_dprime": T_d, "transfer_rho": T_r,
           "cos": cos, "cos_null95_user_asst": null95, "rho_asst_along_vuser": rho_par,
           "rho_asst_orthogonal": rho_orth, "verdict": verdict, "per_layer": per_layer,
           "smoke": bool(a.smoke)}
    (out / "report_g8b.json").write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {out / 'report_g8b.json'}")


if __name__ == "__main__":
    main()
