"""CPU-only stats on the existing full-FT 3-fold result (no GPU, no training):
(1) bootstrap 95% CI on the full-FT trained-verb mean (is 0.823 tight?);
(2) paired bootstrap on (full-FT - head-only baseline) over the SAME fold-0/1/2 objects (is the +0.17 significant?).
The +0.17 is the CONFOUNDED gain (full-FT vs head-only) -- not yet decomposed into SLAT-encoder vs trunk-unfreezing."""
import json, collections, numpy as np
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
rng = np.random.default_rng(0)

ft = json.load(open("/tmp/slat_cv_ft3.json"))          # (verb, obj, ap) full-FT, folds 0/1/2
anchor = json.load(open("/tmp/slat_anchor.json"))       # (verb, obj, ap_base_headonly, ap_slat, ap_ens) all folds

# full-FT per (verb,obj); head-only baseline per (verb,obj) restricted to objects the full-FT covers
ftm = {(v, o): a for v, o, a in ft}
base = {(v, o): ab for v, o, ab, *_ in anchor}
objs = sorted({o for _, o in ftm})                      # fold-0/1/2 objects (the full-FT set)

def trained_mean(pairs, tbl):
    byv = collections.defaultdict(list)
    for v, o in pairs:
        if (v, o) in tbl: byv[v].append(tbl[(v, o)])
    ms = [np.mean(byv[v]) for v in TRAINED if byv[v]]
    return np.mean(ms) if ms else np.nan

allpairs = list(ftm.keys())
pt_ft = trained_mean(allpairs, ftm)
paired = [(v, o) for (v, o) in allpairs if (v, o) in base]     # objects with BOTH scores
pt_base = trained_mean(paired, base)
pt_ftp = trained_mean(paired, ftm)

# bootstrap over objects
B = 4000
ci_ft = np.empty(B); gap = np.empty(B)
byobj_ft = collections.defaultdict(list); byobj_pair = collections.defaultdict(list)
for v, o in allpairs: byobj_ft[o].append((v, o))
for v, o in paired: byobj_pair[o].append((v, o))
pair_objs = sorted(byobj_pair)
for i in range(B):
    draw = rng.choice(objs, len(objs), replace=True)
    pairs = [p for o in draw for p in byobj_ft[o]]
    ci_ft[i] = trained_mean(pairs, ftm)
    draw2 = rng.choice(pair_objs, len(pair_objs), replace=True)
    pp = [p for o in draw2 for p in byobj_pair[o]]
    gap[i] = trained_mean(pp, ftm) - trained_mean(pp, base)

lo, hi = np.percentile(ci_ft, [2.5, 97.5])
glo, ghi = np.percentile(gap, [2.5, 97.5])
print(f"full-FT trained-mean = {pt_ft:.3f}   bootstrap 95% CI = [{lo:.3f}, {hi:.3f}]  (n_obj={len(objs)})")
print(f"head-only baseline (same objs) = {pt_base:.3f} | full-FT (same objs) = {pt_ftp:.3f}")
print(f"CONFOUNDED gain (full-FT - head-only) = {pt_ftp-pt_base:+.3f}   95% CI = [{glo:+.3f}, {ghi:+.3f}]  P(gain>0) = {(gap>0).mean():.3f}")
print("NOTE: this gain is NOT yet split into SLAT-encoder vs trunk-unfreezing -- that is the pending confound check.")
