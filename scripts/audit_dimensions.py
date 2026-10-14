"""Measure actual evaluated geometry in meters; never infer scale from display labels."""
import bpy,json,math
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
s=bpy.data.scenes['01 • HEADLINER TRANSFER'];bpy.context.window.scene=s;s.frame_set(1);bpy.context.view_layer.update()
assert all(sc.unit_settings.system=='METRIC' and sc.unit_settings.scale_length==1 for sc in bpy.data.scenes)
assert max(abs(v-1) for o in bpy.data.objects for v in o.scale)<1e-6
p=bpy.data.objects['HEADLINER • midsurface only']
deps=bpy.context.evaluated_depsgraph_get()
def local_mesh(o):
 ev=o.evaluated_get(deps);m=ev.to_mesh();verts=[v.co.copy() for v in m.vertices];ev.to_mesh_clear();return verts
pv=local_mesh(p)
dims=[max(v[k] for v in pv)-min(v[k] for v in pv) for k in range(3)]
# Identify and measure each boundary component directly on the control surface.
from collections import Counter,defaultdict
edges=Counter()
for f in p.data.polygons:
 for a,b in zip(f.vertices,(*f.vertices[1:],f.vertices[0])):edges[tuple(sorted((a,b)))]+=1
adj=defaultdict(set)
for (a,b),n in edges.items():
 if n==1:adj[a].add(b);adj[b].add(a)
remain=set(adj);loops=[]
while remain:
 stack=[next(iter(remain))];ids=[]
 while stack:
  v=stack.pop()
  if v not in remain:continue
  remain.remove(v);ids.append(v);stack.extend(adj[v])
 vs=[p.data.vertices[i].co for i in ids]
 loops.append({'width_m':max(v.x for v in vs)-min(v.x for v in vs),'length_m':max(v.y for v in vs)-min(v.y for v in vs),'center_m':[sum(v[k] for v in vs)/len(vs) for k in range(3)]})
loops.sort(key=lambda b:b['width_m']*b['length_m'],reverse=True)
shoulder=bpy.data.objects['J2 • shoulder elevation'];elbow=bpy.data.objects['J3 • elbow flexion'];wrist=bpy.data.objects['J5 • wrist level compensation']
links=[(elbow.matrix_world.translation-shoulder.matrix_world.translation).length,(wrist.matrix_world.translation-elbow.matrix_world.translation).length]
cup=bpy.data.objects['Cup 1 • open bellows'];lip=cup.data.vertices[:48]
lip_diameter=2*max(math.hypot(v.co.x,v.co.y) for v in lip)
hose=bpy.data.objects['Vacuum hose 1']
rod=bpy.data.objects['SUCTION 01 • contact • adjustable plunger']
pad=bpy.data.objects['Soft nest pad']
assert 1.3<dims[0]<1.5 and 1.8<dims[1]<2.0 and .08<dims[2]<.25
assert abs(lip_diameter-.100)<.002
assert abs(hose.data.bevel_depth*2-.010)<1e-7
assert 3.1<sum(links)<3.3
records=json.loads((ROOT/'validation/motion.json').read_text())
report={'units':'meters','meters_per_blender_unit':1.0,'all_scenes_metric':True,'all_object_scales_applied':True,
'headliner_evaluated_width_length_depth_m':dims,'headliner_nominal_physical_thickness_m':p['physical_thickness_m'],
'headliner_has_thickness_geometry':False,'boundary_dimensions_control_mesh':loops,
'robot_main_link_lengths_m':links,'robot_main_link_sum_m':sum(links),
'station_center_spacing_m':max(r['headliner_center'][0] for r in records)-min(r['headliner_center'][0] for r in records),
'headliner_origin_station_height_m':min(r['headliner_center'][2] for r in records),
'headliner_origin_peak_height_m':max(r['headliner_center'][2] for r in records),
'suction_lip_nominal_diameter_m':.1,'suction_lip_measured_diameter_m':lip_diameter,
'suction_bellows_nominal_height_m':.061,'vacuum_hose_outside_diameter_m':hose.data.bevel_depth*2,
'plunger_rod_diameter_m':2*max(math.hypot(v.co.x,v.co.y) for v in rod.data.vertices),
'fixture_pad_diameter_m':2*max(math.hypot(v.co.x,v.co.y) for v in pad.data.vertices),
'joint_cover_fastener_nominal_diameter_m':.013,'anchor_bolt_head_nominal_diameter_m':.036,
'design_status':'Representative passenger-car panel and long-reach industrial arm. Not measured OEM CAD; nominal shell thickness remains metadata only.'}
(ROOT/'validation/dimensions.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
