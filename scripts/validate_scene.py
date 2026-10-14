"""Audit the saved asset, including topology, rigid motion, and arm/shell intersections."""
import bpy, json
from pathlib import Path
from collections import Counter, defaultdict
from mathutils import Vector
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[1]
s=bpy.data.scenes['01 • HEADLINER TRANSFER'];bpy.context.window.scene=s
p=bpy.data.objects['HEADLINER • midsurface only'];mesh=p.data
assert all(len(f.vertices)==4 for f in mesh.polygons)
assert len(p.modifiers)==1 and p.modifiers[0].type=='SUBSURF'
adj=defaultdict(set);counts=Counter()
for f in mesh.polygons:
 for a,b in zip(f.vertices,(*f.vertices[1:],f.vertices[0])):
  adj[a].add(b);adj[b].add(a);counts[tuple(sorted((a,b)))]+=1
seen=set();stack=[0]
while stack:
 v=stack.pop()
 if v in seen:continue
 seen.add(v);stack.extend(adj[v]-seen)
assert len(seen)==len(mesh.vertices),'Disconnected headliner'
boundary=defaultdict(set)
for (a,b),n in counts.items():
 assert n in [1,2]
 if n==1:boundary[a].add(b);boundary[b].add(a)
assert all(len(v)==2 for v in boundary.values())
remaining=set(boundary);loops=0
while remaining:
 loops+=1;stack=[next(iter(remaining))]
 while stack:
  v=stack.pop()
  if v not in remaining:continue
  remaining.remove(v);stack.extend(boundary[v])
assert loops==6
cups=[bpy.data.objects[f'SUCTION {i:02d} • contact'] for i in range(1,5)]
parts=[o for o in bpy.data.collections['01 • Robot — rigid links'].objects if o.type=='MESH' and ('rigid casting' in o.name or 'gearbox' in o.name)]
expected=json.loads((ROOT/'validation/motion.json').read_text())
def tree(o,deps):
 ev=o.evaluated_get(deps);m=ev.to_mesh()
 t=BVHTree.FromPolygons([o.matrix_world@v.co for v in m.vertices],[tuple(f.vertices) for f in m.polygons]);ev.to_mesh_clear();return t
max_error=0;collisions=[];max_contact=0;max_scale_error=0
for f in range(1,193):
 s.frame_set(f);bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
 max_error=max(max_error,(p.matrix_world.translation-Vector(expected[f-1]['headliner_center'])).length)
 surf=tree(p,deps)
 for cup in cups:max_contact=max(max_contact,surf.find_nearest(cup.matrix_world.translation)[3])
 for part in parts:
  max_scale_error=max(max_scale_error,max(abs(v-1) for v in part.matrix_world.to_scale()))
  if surf.overlap(tree(part,deps)):collisions.append([f,part.name])
assert max_error<1e-5 and max_contact<1e-5 and max_scale_error<1e-5
assert not collisions,collisions[:10]
report={'saved_scene_audited':True,'connected_surface_components':1,'open_boundary_loops':loops,'all_control_faces_quads':True,'frames_checked':192,'suction_contacts_checked_per_frame':4,'maximum_suction_center_error_m':max_contact,'maximum_path_error_m':max_error,'maximum_arm_scale_error':max_scale_error,'arm_shell_surface_intersections':collisions,'collision_check_scope':'Evaluated headliner versus upper-arm/forearm castings (including covers/ribs) and joint gearbox meshes at each integer frame. Not comprehensive cell or self-collision certification.'}
(ROOT/'validation/asset-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
