# GEAL teacher fix for transformers ≥ 5 (batch_encode_plus removed)

`external/geal` is a vendored clone (gitignored), so this fix lives only on disk in this checkout.
**Re-apply it after any fresh GEAL clone**, or GEAL pseudolabel generation fails with:

```
AttributeError('RobertaTokenizer has no attribute batch_encode_plus')
```

`batch_encode_plus` was removed in `transformers` 5.x; the tokenizer's `__call__` is the drop-in
equivalent (same `padding` / `truncation` / `max_length` / `return_tensors` kwargs).

## The change (2 files)

`external/geal/model/branch_3d.py` (~line 190) and `external/geal/model/branch_2d.py` (~line 230):

```python
-        tokens = self.tokenizer.batch_encode_plus(
+        tokens = self.tokenizer(
             text_queries,            # (branch_2d.py: `queries`)
             padding='max_length',
             truncation=True,
             max_length=self.n_groups,
             return_tensors='pt'
         ).to(device)
```

One-liner to apply to a fresh clone:

```bash
sed -i 's/self\.tokenizer\.batch_encode_plus(/self.tokenizer(/' \
  external/geal/model/branch_3d.py external/geal/model/branch_2d.py
```
