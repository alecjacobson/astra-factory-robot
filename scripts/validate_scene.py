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
# No inverted or collapsed triangles in the midsurface's XY projection.
min_projected_triangle_area=float('inf')
for face in mesh.polygons:
 a,b,c,d=[mesh.vertices[i].co for i in face.vertices]
 for area in [(b-a).cross(c-a).z/2,(c-a).cross(d-a).z/2]:
  min_projected_triangle_area=min(min_projected_triangle_area,area)
assert min_projected_triangle_area>1e-10
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
assert loops==p['open_apertures']+1
cups=[bpy.data.objects[f'SUCTION {i:02d} • contact'] for i in range(1,5)]
parts=[o for o in bpy.data.collections['01 • Robot — rigid links'].objects if o.type=='MESH' and ('rigid casting' in o.name or 'gearbox' in o.name)]
expected=json.loads((ROOT/'validation/motion.json').read_text())
# Cache evaluated rigid geometry once, then transform arm meshes into panel space.
# This avoids rebuilding the 130k-quad subdivided shell on every frame.
s.frame_set(1);bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
def geometry(o):
 ev=o.evaluated_get(deps);m=ev.to_mesh();verts=[v.co.copy() for v in m.vertices];faces=[tuple(f.vertices) for f in m.polygons];ev.to_mesh_clear();return verts,faces
verts,faces=geometry(p);surf=BVHTree.FromPolygons(verts,faces)
cached={part:geometry(part) for part in parts}
max_error=0;collisions=[];max_contact=0;max_scale_error=0;max_lip_error=0;max_pad_error=0
for f in range(1,193):
 s.frame_set(f);bpy.context.view_layer.update();inv=p.matrix_world.inverted()
 max_error=max(max_error,(p.matrix_world.translation-Vector(expected[f-1]['headliner_center'])).length)
 for cup in cups:
  max_contact=max(max_contact,surf.find_nearest(inv@cup.matrix_world.translation)[3])
  bellows=next(o for o in cup.children if 'open bellows' in o.name)
  transform=inv@bellows.matrix_world
  for vertex in bellows.data.vertices[:48]:max_lip_error=max(max_lip_error,surf.find_nearest(transform@vertex.co)[3])
 for part,(pv,pf) in cached.items():
  max_scale_error=max(max_scale_error,max(abs(v-1) for v in part.matrix_world.to_scale()))
  transform=inv@part.matrix_world
  if surf.overlap(BVHTree.FromPolygons([transform@v for v in pv],pf)):collisions.append([f,part.name])
 if f in [1,97]:
  center=p.matrix_world.translation
  for pad in (o for o in s.objects if o.name.startswith('Soft nest pad') and abs(o.location.x-center.x)<.7):
   top=pad.matrix_world@Vector((0,0,.035*.85/2))
   max_pad_error=max(max_pad_error,surf.find_nearest(inv@top)[3])
assert max_error<1e-5 and max_contact<1e-5 and max_scale_error<1e-5
assert max_lip_error<2e-5 and max_pad_error<2e-5
assert not collisions,collisions[:10]
report={'saved_scene_audited':True,'connected_surface_components':1,'open_boundary_loops':loops,'all_control_faces_quads':True,'minimum_projected_triangle_area_m2':min_projected_triangle_area,'frames_checked':192,'suction_contacts_checked_per_frame':4,'maximum_suction_center_error_m':max_contact,'maximum_path_error_m':max_error,'maximum_arm_scale_error':max_scale_error,'maximum_suction_lip_surface_error_m':max_lip_error,'maximum_fixture_pad_contact_error_m':max_pad_error,'arm_shell_surface_intersections':collisions,'collision_check_scope':'Evaluated headliner versus upper-arm/forearm castings (including covers/ribs) and joint gearbox meshes at each integer frame. Not comprehensive cell or self-collision certification.'}
(ROOT/'validation/asset-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
