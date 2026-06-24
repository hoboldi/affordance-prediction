"""Minimal 3D affordance ground-truth painter (human-only; no SAM, no projection).

Loads each object's mesh.glb in the SAME vertex order the model uses (datasets.mesh_loading.load_mesh,
process=False), serves it to a browser three.js viewer, lets you orbit + brush-paint the affordance region
for a chosen verb directly on the 3D surface, and saves per-vertex binary labels (native order) to
``gt_labels/<sample_id>__<verb>.pt`` for an honest, pipeline-independent eval set.

Run:  PYTHONPATH=src python scripts/mesh_painter.py --port 8765
Then open http://localhost:8765  (forward the port if remote).
"""
from __future__ import annotations
import argparse, base64, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import torch
from datasets.data_root_dataset import DataRootDataset
from datasets.mesh_loading import load_mesh

VERBS = ["grasp", "contain", "pour", "sit", "move"]
OBJECTS: list[dict] = []      # [{name, mesh_path}]
GT_DIR = _REPO / "gt_labels"

HTML = r"""<!doctype html><html><head><meta charset=utf-8><title>Affordance painter</title>
<style>body{margin:0;font-family:sans-serif;overflow:hidden}#bar{position:fixed;top:0;left:0;right:0;
background:#222;color:#eee;padding:6px 10px;z-index:10;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
#bar select,#bar button{font-size:14px;padding:3px 6px}#bar .tag{color:#9f9}#c{position:absolute;top:0;left:0}
#status{margin-left:auto;color:#9cf}</style></head><body>
<div id=bar>
 <span>Obj</span><select id=obj></select>
 <span>Verb</span><select id=verb></select>
 <label>brush <input id=brush type=range min=0.005 max=0.15 step=0.005 value=0.04></label>
 <button id=mode>Paint</button><button id=clear>Clear verb</button>
 <button id=save>Save</button><span class=tag id=painted></span><span id=status></span>
</div>
<canvas id=c></canvas>
<script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
<script>
const $=id=>document.getElementById(id);
let renderer,scene,cam,controls,mesh,geom,labels,nV,erasing=false,painting=false,curObj=0;
const status=m=>$("status").textContent=m;
function initGL(){
 renderer=new THREE.WebGLRenderer({canvas:$("c"),antialias:true});
 renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(devicePixelRatio);
 scene=new THREE.Scene();scene.background=new THREE.Color(0x202428);
 cam=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.01,100);cam.position.set(0,0,3);
 controls=new THREE.OrbitControls(cam,renderer.domElement);
 scene.add(new THREE.AmbientLight(0xffffff,0.7));
 const d=new THREE.DirectionalLight(0xffffff,0.8);d.position.set(1,1,2);scene.add(d);
 addEventListener("resize",()=>{renderer.setSize(innerWidth,innerHeight);cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();});
 const ray=new THREE.Raycaster(),m=new THREE.Vector2();
 function paintAt(e){
  if(!mesh)return;m.x=e.clientX/innerWidth*2-1;m.y=-(e.clientY/innerHeight)*2+1;
  ray.setFromCamera(m,cam);const hit=ray.intersectObject(mesh)[0];if(!hit)return;
  const p=hit.point,r=parseFloat($("brush").value),r2=r*r,pos=geom.attributes.position.array,col=geom.attributes.color.array;
  for(let i=0;i<nV;i++){const dx=pos[3*i]-p.x,dy=pos[3*i+1]-p.y,dz=pos[3*i+2]-p.z;
   if(dx*dx+dy*dy+dz*dz<r2){labels[i]=erasing?0:1;}}
  recolor();
 }
 renderer.domElement.addEventListener("pointerdown",e=>{if(e.shiftKey){painting=true;controls.enabled=false;paintAt(e);}});
 renderer.domElement.addEventListener("pointermove",e=>{if(painting)paintAt(e);});
 addEventListener("pointerup",()=>{painting=false;controls.enabled=true;});
 (function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,cam);})();
}
function recolor(){
 const col=geom.attributes.color.array;let n=0;
 for(let i=0;i<nV;i++){if(labels[i]){col[3*i]=0.95;col[3*i+1]=0.15;col[3*i+2]=0.1;n++;}
  else{col[3*i]=0.75;col[3*i+1]=0.75;col[3*i+2]=0.78;}}
 geom.attributes.color.needsUpdate=true;$("painted").textContent=n+" / "+nV+" verts";
}
function b64f32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}
function b64u32(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Uint32Array(u.buffer);}
function b64u8(s){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}
async function loadMesh(){
 status("loading...");const r=await fetch("/api/mesh?i="+curObj+"&verb="+$("verb").value);const d=await r.json();
 const verts=b64f32(d.vertices),faces=b64u32(d.faces);nV=d.n_verts;labels=Array.from(b64u8(d.labels));
 if(mesh){scene.remove(mesh);geom.dispose();}
 geom=new THREE.BufferGeometry();geom.setAttribute("position",new THREE.BufferAttribute(verts,3));
 geom.setIndex(new THREE.BufferAttribute(faces,1));
 geom.setAttribute("color",new THREE.BufferAttribute(new Float32Array(nV*3),3));geom.computeVertexNormals();
 mesh=new THREE.Mesh(geom,new THREE.MeshStandardMaterial({vertexColors:true,roughness:0.9,side:THREE.DoubleSide}));
 // center + scale to fit
 geom.computeBoundingSphere();const bs=geom.boundingSphere;mesh.position.set(-bs.center.x,-bs.center.y,-bs.center.z);
 cam.position.set(0,0,bs.radius*3);controls.target.set(0,0,0);controls.update();
 scene.add(mesh);recolor();status("Shift+drag to paint. "+d.name);
}
async function save(){
 status("saving...");
 await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({i:curObj,verb:$("verb").value,labels:labels})});
 status("saved "+$("verb").value);
}
$("mode").onclick=()=>{erasing=!erasing;$("mode").textContent=erasing?"Erase":"Paint";};
$("clear").onclick=()=>{labels=labels.map(()=>0);recolor();};
$("save").onclick=save;$("obj").onchange=()=>{curObj=+$("obj").value;loadMesh();};
$("verb").onchange=loadMesh;
async function start(){
 initGL();const d=await(await fetch("/api/objects")).json();
 d.verbs.forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;$("verb").appendChild(o);});
 d.objects.forEach((o,i)=>{const e=document.createElement("option");e.value=i;e.textContent=o;$("obj").appendChild(e);});
 await loadMesh();
}
start();
</script></body></html>"""


def _b64(arr) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, ctype, body):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/":
            self._send(200, "text/html", HTML.encode())
        elif u.path == "/api/objects":
            self._send(200, "application/json", json.dumps({"objects": [o["name"] for o in OBJECTS], "verbs": VERBS}).encode())
        elif u.path == "/api/mesh":
            i = int(q.get("i", ["0"])[0]); verb = q.get("verb", [VERBS[0]])[0]; o = OBJECTS[i]
            m = load_mesh(o["mesh_path"], process=False)
            v = np.asarray(m.vertices, np.float32); f = np.asarray(m.faces, np.uint32).reshape(-1)
            gt = GT_DIR / f"{o['name']}__{verb}.pt"
            lab = (torch.load(gt, map_location="cpu")["labels"].numpy().astype(np.uint8)
                   if gt.is_file() else np.zeros(len(v), np.uint8))
            self._send(200, "application/json", json.dumps({
                "n_verts": int(len(v)), "name": o["name"],
                "vertices": _b64(v), "faces": _b64(f), "labels": _b64(lab)}).encode())
        else:
            self._send(404, "text/plain", b"nope")

    def do_POST(self):
        if self.path != "/api/save":
            self._send(404, "text/plain", b"nope"); return
        n = int(self.headers.get("Content-Length", 0)); d = json.loads(self.rfile.read(n))
        o = OBJECTS[int(d["i"])]; verb = d["verb"]; lab = torch.tensor(d["labels"], dtype=torch.uint8)
        GT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"labels": lab, "verb": verb, "name": o["name"], "n_pos": int(lab.sum())},
                   GT_DIR / f"{o['name']}__{verb}.pt")
        print(f"saved {o['name']}__{verb}: {int(lab.sum())} positive verts")
        self._send(200, "application/json", b'{"ok":true}')


def main():
    global GT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--manifest", default="/home/datasets/customDatasets/cmr2/manifest.pseudolabeled.clean.jsonl")
    ap.add_argument("--gt_dir", default=str(GT_DIR))
    ap.add_argument("--limit", type=int, default=60, help="number of objects to expose for labelling")
    args = ap.parse_args()
    GT_DIR = Path(args.gt_dir)
    ds = DataRootDataset(manifest_path=args.manifest, load_vertex_labels_eager=False, load_vertex_semantics_eager=False)
    seen = set()
    for r in ds.rows:
        d = r.sam3d_reconstruction_dir
        if d is None or str(d) in seen:
            continue
        mp = d / "mesh.glb"
        if mp.is_file():
            seen.add(str(d)); OBJECTS.append({"name": d.name, "mesh_path": mp})
        if len(OBJECTS) >= args.limit:
            break
    print(f"Serving {len(OBJECTS)} objects on http://localhost:{args.port}  (GT -> {GT_DIR})")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
