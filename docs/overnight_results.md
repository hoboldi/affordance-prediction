# Overnight results — started Tue Jun 23 10:00:41 PM UTC 2026
### STAGE0 v13_dinolarge  22:33
  contain     AUPRC=0.177 (n= 39)  pred_std=0.0802
  grasp       AUPRC=0.221 (n= 29)  pred_std=0.1896
  move        AUPRC=0.580 (n= 10)  pred_std=0.1666
  pour        AUPRC=0.562 (n= 29)  pred_std=0.1803
  sit         AUPRC=0.509 (n= 10)  pred_std=0.2037
VERDICT corr_learn=-0.419 iou_learn=0.000 std=0.1501

### STAGE0 v15_ovnc  22:35
  contain     AUPRC=0.322 (n= 39)  pred_std=0.1615
  grasp       AUPRC=0.339 (n= 29)  pred_std=0.1570
  move        AUPRC=0.521 (n= 10)  pred_std=0.1663
  pour        AUPRC=0.522 (n= 29)  pred_std=0.1800
  sit         AUPRC=0.414 (n= 10)  pred_std=0.1931
VERDICT corr_learn=0.006 iou_learn=0.000 std=0.1680

### base_l0  (--contrastive_weight 0)  02:53
  contain     AUPRC=0.297 (n= 39)  pred_std=0.1386
  grasp       AUPRC=0.360 (n= 29)  pred_std=0.1445
  move        AUPRC=0.551 (n= 10)  pred_std=0.1793
  pour        AUPRC=0.609 (n= 29)  pred_std=0.1832
  sit         AUPRC=0.477 (n= 10)  pred_std=0.2009
VERDICT corr_learn=-0.338 iou_learn=0.000 std=0.1598

### base_l03  (--contrastive_weight 0.3)  04:14
  contain     AUPRC=0.228 (n= 39)  pred_std=0.1074
  grasp       AUPRC=0.278 (n= 29)  pred_std=0.1266
  move        AUPRC=0.575 (n= 10)  pred_std=0.1712
  pour        AUPRC=0.610 (n= 29)  pred_std=0.1864
  sit         AUPRC=0.503 (n= 10)  pred_std=0.1894
VERDICT corr_learn=-0.428 iou_learn=0.000 std=0.1441

### base_crossattn_nocontrast (cross_attn, no-contrastive) 06:36
### cross-attn (stopped after ep20) — best.pt PER-VERB  08:39
  contain     AUPRC=0.302 (n= 39)  pred_std=0.1557
  grasp       AUPRC=0.382 (n= 29)  pred_std=0.1499
  move        AUPRC=0.602 (n= 10)  pred_std=0.1639
  pour        AUPRC=0.646 (n= 29)  pred_std=0.1846
  sit         AUPRC=0.490 (n= 10)  pred_std=0.1963
VERDICT corr_learn=-0.500 iou_learn=0.000 std=0.1655

  contain     AUPRC=0.301 (n= 39)  pred_std=0.1365
  grasp       AUPRC=0.370 (n= 29)  pred_std=0.1098
  move        AUPRC=0.590 (n= 10)  pred_std=0.1405
  pour        AUPRC=0.630 (n= 29)  pred_std=0.1805
  sit         AUPRC=0.475 (n= 10)  pred_std=0.2030
VERDICT corr_learn=-0.463 iou_learn=0.000 std=0.1467
### cross-attn (stopped after ep20) — last.pt PER-VERB
  contain     AUPRC=0.302 (n= 39)  pred_std=0.1557
  grasp       AUPRC=0.382 (n= 29)  pred_std=0.1499
  move        AUPRC=0.602 (n= 10)  pred_std=0.1639
  pour        AUPRC=0.646 (n= 29)  pred_std=0.1846
  sit         AUPRC=0.490 (n= 10)  pred_std=0.1963
VERDICT corr_learn=-0.500 iou_learn=0.000 std=0.1655

### ov_concat_nc (open-vocab concat, flagship) 16:41
  contain     AUPRC=0.308 (n= 39)  pred_std=0.1508
  grasp       AUPRC=0.305 (n= 29)  pred_std=0.1806
  move        AUPRC=0.397 (n= 10)  pred_std=0.1887
  pour        AUPRC=0.538 (n= 29)  pred_std=0.1851
  sit         AUPRC=0.420 (n= 10)  pred_std=0.1879
VERDICT corr_learn=0.036 iou_learn=0.000 std=0.1731

### ov_concat_bighead (open-vocab concat + deep_verb_proj + 512/256/128) 14:37
