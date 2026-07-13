"""Compare original (25-60 upper band) vs wideband (lower-mirror added) per-vertex visibility on the
prototype objects, and how many grasp human-positives that were invisible become visible. CPU."""
import os, json, numpy as np, torch
ROOT = "/home/datasets/customDatasets/cmr2"
protos = [json.loads(l)["sam3d_reconstruction_dir"].split("/")[-1] for l in open("/tmp/tier2_proto.jsonl")]
print(f"{'object':38s} {'orig_vis%':>9s} {'wide_vis%':>9s}   grasp+ recovery")
g_inv_tot = g_now_tot = 0
for o in protos:
    rd = f"{ROOT}/reconstructions/{o}"
    cw_path = f"{rd}/vertex_semantics_wideband.pt"
    if not os.path.exists(cw_path):
        print(f"{o:38s}   (no wideband file)"); continue
    vo = np.asarray(torch.load(f"{rd}/vertex_semantics_clean.pt", weights_only=False)["visible_in_any_view"]).astype(bool)
    vw = np.asarray(torch.load(cw_path, weights_only=False)["visible_in_any_view"]).astype(bool)
    line = f"{o:38s} {vo.mean()*100:9.1f} {vw.mean()*100:9.1f}"
    hf = f"human_gt_labels/{o}/vertex_manuallabels_grasp.pt"
    if os.path.exists(hf):
        hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5)
        if hb.sum() and len(hb) == len(vo) == len(vw):
            inv = int((hb & ~vo).sum()); now = int((hb & ~vo & vw).sum())
            g_inv_tot += inv; g_now_tot += now
            line += f"   grasp+ invis(orig)={inv:5d} -> now visible={now:5d} ({now/max(1,inv)*100:3.0f}%)"
    print(line)
if g_inv_tot:
    print(f"\nTOTAL grasp+ recovery: {g_now_tot}/{g_inv_tot} = {g_now_tot/g_inv_tot*100:.0f}% of previously-invisible grasp positives now visible with wideband")
