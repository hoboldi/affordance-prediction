"""Reserve most of GPU0's free VRAM so other users can't start a job while we run our sweep.
Leaves `margin_gb` free for OUR training jobs. TIME-BOXED: kill this pid when our experiments finish.
Run with CUDA_VISIBLE_DEVICES=0."""
import torch, time, os, sys
margin_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 9.0
blocks = []
while True:
    free, total = torch.cuda.mem_get_info(0)
    if free / 1e9 <= margin_gb:
        break
    take_gb = min(1.0, free / 1e9 - margin_gb)
    try:
        blocks.append(torch.empty(int(take_gb * 1e9 / 4), dtype=torch.float32, device="cuda:0"))
    except RuntimeError:
        break
free, total = torch.cuda.mem_get_info(0)
print(f"HOLDER pid={os.getpid()} active — holding {sum(b.numel()*4 for b in blocks)/1e9:.1f} GB, "
      f"free now {free/1e9:.1f}/{total/1e9:.1f} GB (margin {margin_gb} GB left for our jobs)", flush=True)
while True:
    time.sleep(300)
