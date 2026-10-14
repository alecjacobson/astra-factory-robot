"""Deterministic Blender 4.5 scene builder; run with blender -b --python scripts/build_scene.py."""
import bpy, math, json, os, sys
from pathlib import Path
from mathutils import Vector
from collections import Counter, defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from headliner_design import DESIGN_SCALE, CUP_DIAMETER_M, HOSE_OUTSIDE_DIAMETER_M, APERTURES, height, build_mesh
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
for c in list(bpy.data.collections):
    if c.name!='Collection': bpy.data.collections.remove(c)
scene=bpy.context.scene; scene.name='01 • HEADLINER TRANSFER'
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1.0;scene.unit_settings.length_unit='METERS'
basecoll=bpy.data.collections.get('Collection');basecoll.name='00 • Cell architecture'
def collection(name):
 c=bpy.data.collections.new(name);scene.collection.children.link(c);return c
robotcoll=collection('01 • Robot — rigid links');toolcoll=collection('02 • Vacuum tooling — configurable contacts');panelcoll=collection('03 • Headliner — open midsurface');stagecoll=basecoll
activecoll=stagecoll
def move(o):
 for c in list(o.users_collection):c.objects.unlink(o)
 activecoll.objects.link(o)
 return o
def mat(name,color,metal=0,rough=.4):
 m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
 p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Metallic'].default_value=metal;p.inputs['Roughness'].default_value=rough
 return m
orange=mat('Robot | signal orange enamel',(.94,.205,.035),.5,.28)
dark=mat('Graphite | cast housings',(.035,.049,.062),.65,.3)
steel=mat('Machined aluminum',(.48,.60,.65),.8,.25)
black=mat('Vacuum elastomer',(.018,.024,.028),.05,.6)
teal=mat('Vacuum circuit | turquoise',(.025,.63,.59),.35,.3)
cream=mat('Headliner | warm ivory textile',(.72,.66,.51),0,.8)
# Fine procedural textile, no external assets.
p=cream.node_tree.nodes.get('Principled BSDF');noise=cream.node_tree.nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=2400
coord=cream.node_tree.nodes.new('ShaderNodeTexCoord');cream.node_tree.links.new(coord.outputs['Object'],noise.inputs['Vector'])
p.inputs['Sheen Weight'].default_value=.18
bump=cream.node_tree.nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.17;bump.inputs['Distance'].default_value=.00012
cream.node_tree.links.new(noise.outputs['Fac'],bump.inputs['Height']);cream.node_tree.links.new(bump.outputs['Normal'],p.inputs['Normal'])
floor=mat('Cell | blue charcoal',(.052,.075,.092),.25,.5)
white=mat('Markings | pearl',(.69,.79,.81),.1,.5)
yellow=mat('Safety | ochre',(.95,.54,.07),.1,.5)
def finish(o,name,material,parent=None):
 o.name=name;move(o)
 if material:o.data.materials.append(material)
 if parent:o.parent=parent
 return o
def bevel(o,w=.035):
 m=o.modifiers.new('Manufactured edge radii','BEVEL');m.width=w;m.segments=3
 o.modifiers.new('Weighted corner normals','WEIGHTED_NORMAL')
 return o
def box(name,loc,dim,material,parent=None,bev=.025):
 bpy.ops.mesh.primitive_cube_add(size=1,location=loc);o=finish(bpy.context.object,name,material,parent);o.dimensions=dim
 bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
 if bev:bevel(o,bev)
 return o
def cyl(name,loc,r,depth,material,parent=None,axis='Z',vertices=48):
 bpy.ops.mesh.primitive_cylinder_add(vertices=vertices,radius=r,depth=depth,location=loc);o=finish(bpy.context.object,name,material,parent)
 if axis=='Y':o.rotation_euler[0]=math.pi/2
 if axis=='X':o.rotation_euler[1]=math.pi/2
 bevel(o,.008)
 for p in o.data.polygons:p.use_smooth=True
 return o
def empty(name,loc=(0,0,0),parent=None):
 o=bpy.data.objects.new(name,None);activecoll.objects.link(o);o.location=loc;o.parent=parent;o.empty_display_size=.18;return o
def curve(name,pts,r,material,parent=None):
 d=bpy.data.curves.new(name,'CURVE');d.dimensions='3D';d.bevel_depth=r;d.bevel_resolution=3
 s=d.splines.new('BEZIER');s.bezier_points.add(len(pts)-1)
 for p,co in zip(s.bezier_points,pts):p.co=co;p.handle_left_type='AUTO';p.handle_right_type='AUTO'
 o=bpy.data.objects.new(name,d);activecoll.objects.link(o);o.parent=parent;d.materials.append(material);return o
def label(name,text,loc,size,material,parent=None,rot=(0,0,0)):
 d=bpy.data.curves.new(name,'FONT');d.body=text;d.size=size;d.extrude=.0008;d.align_x='CENTER'
 o=bpy.data.objects.new(name,d);activecoll.objects.link(o);o.location=loc;o.rotation_euler=rot;o.parent=parent;d.materials.append(material);return o
# Cell fixtures.
box('Raised cell plinth',(0,0,-.15),(8.4,5.7,.3),floor,bev=.16)
box('Studio ground',(0,0,-.36),(200,200,.12),mat('Background',(.021,.031,.043),.05,.7),bev=0)
for x in [-2.35,2.35]:
 box('Station footprint',(x,-.35,.015),(2.05,2.65,.03),dark)
 for dx in [-.66,.66]:
  for dy in [-.87,.87]:
   box('Fixture leg',(x+dx,-.35+dy,.48),(.09,.09,.94),steel)
   cyl('Adjustable foot',(x+dx,-.35+dy,.09),.10,.06,dark)
 for y in [-1.22,.52]:
  box('Nest crossmember',(x,y,.89),(1.52,.11,.13),steel)
 for dx in [-.62,.62]:
  box('Nest longitudinal rail',(x+dx,-.35,.88),(.08,1.94,.14),steel)
  for dy in [-.72,.72]:
   cyl('Locating pedestal',(x+dx,-.35+dy,1.052),.020/DESIGN_SCALE,.224,dark)
   cyl('Soft nest pad',(x+dx,-.35+dy,1.181),.035/DESIGN_SCALE,.035,teal)
 label('Station designation','A  /  INFEED' if x<0 else 'B  /  OUTFEED',(x,-1.80,.04),.19,white)
 for yy in [-1.67,.99]:
  box('Fixture floor marking',(x,yy,.035),(1.95,.022,.008),teal,bev=0)
# Inset perimeter line, hazard tabs.
for y in [-2.55,2.55]:box('Perimeter stripe',(0,y,.015),(7.9,.026,.009),white,bev=0)
for x in [-3.95,3.95]:box('Perimeter stripe',(x,0,.015),(.026,5.1,.009),white,bev=0)
for x in [-.55,-.32,-.09,.14,.37,.60]:
 o=box('Hazard dash',(x,2.0,.025),(.11,.34,.01),yellow,bev=0);o.rotation_euler[2]=-.55
label('Cell name','H L  /  0 4',(0,-2.42,.03),.24,white)
# Robot hierarchy in radial XZ plane; every visual component is rigidly parented.
activecoll=robotcoll
box('Robot anchor plate',(0,1.65,.13),(1.12,1.12,.26),dark,bev=.08)
for x in [-.43,.43]:
 for y in [1.22,2.08]:cyl('Anchor bolt',(x,y,.28),.018/DESIGN_SCALE,.012/DESIGN_SCALE,steel,vertices=6)
cyl('Fixed pedestal',(0,1.65,.48),.42,.55,dark)
cyl('Axis 1 turntable',(0,1.65,.79),.40,.12,orange)
yaw=empty('J1 • base azimuth',(0,1.65,1.15))
box('Shoulder cradle',(0,0,-.19),(.53,.60,.43),orange,yaw,.08)
shoulder=empty('J2 • shoulder elevation',parent=yaw)
L1=1.95;L2=1.8
elbow=empty('J3 • elbow flexion',(L1,0,0),shoulder)
wrist=empty('J5 • wrist level compensation',(L2,0,0),elbow)
tool=empty('J6 • tool heading compensation',parent=wrist)
for root,r,depth in [(shoulder,.30,.70),(elbow,.245,.54),(wrist,.17,.36)]:
 cyl(root.name+' gearbox',(0,0,0),r,depth,orange,root,'Y')
 for side in [-1,1]:
  cyl('Joint cap',(0,side*(depth/2+.015),0),r*.77,.04,dark,root,'Y')
  cyl('Joint axle',(0,side*(depth/2+.04),0),r*.34,.025,steel,root,'Y')
  for a in range(8):
   phi=2*math.pi*a/8
   cyl('Joint cap fastener',(r*.60*math.cos(phi),side*(depth/2+.041),r*.60*math.sin(phi)),.0065/DESIGN_SCALE,.006/DESIGN_SCALE,steel,root,'Y',12)
def link(name,length,w0,w1,parent):
 # Tapered rigid cast link with planar side panels.
 verts=[]
 for x,w,h in [(.05,w0,w0*.85),(length-.06,w1,w1*.85)]:
  verts.extend([(x,-w,-h),(x,w,-h),(x,w,h),(x,-w,h)])
 faces=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
 me=bpy.data.meshes.new(name);me.from_pydata(verts,[],faces);me.update();o=bpy.data.objects.new(name,me);activecoll.objects.link(o);o.parent=parent;me.materials.append(orange);bevel(o,.07)
 box(name+' service cover',(length*.49,-w0-.008,0),(length*.57,.035,w0*.92),dark,parent,.035)
 box(name+' raised rib',(length*.49,0,w0*.76),(length*.65,.10,.055),orange,parent)
 curve(name+' energy conduit',[(.10,w0*.96,.22),(length*.3,w0*1.32,.28),(length*.7,w1*1.50,.23),(length-.07,w1,.18)],.037,black,parent)
link('Upper arm | rigid casting',L1,.22,.16,shoulder)
link('Forearm | rigid casting',L2,.18,.105,elbow)
label('Robot badge','HL-04',(L1*.5,-.244,.015),.12,white,shoulder,(math.pi/2,0,0))
# Tool flange and suction frame.
activecoll=toolcoll
cyl('Wrist output flange',(0,0,-.09),.145,.18,dark,tool)
cyl('Tool adapter',(0,0,-.20),.19,.055,steel,tool)
for x in [-.56,.56]:box('Vacuum frame longitudinal',(x,0,-.27),(.055,1.53,.075),steel,tool,.012)
for y in [-.65,.65]:box('Vacuum frame crossbar',(0,y,-.27),(1.25,.065,.075),steel,tool,.012)
box('Central tool bridge',(0,0,-.25),(1.19,.11,.085),steel,tool)
box('Vacuum manifold',(0,0,-.15),(.29,.17,.13),dark,tool)
# Refined forming surface: every detail belongs to the same open quad midsurface.
activecoll=panelcoll
verts,faces=build_mesh()
me=bpy.data.meshes.new('Headliner quad control surface • 11 open apertures');me.from_pydata(verts,[],faces);me.update()
panel=bpy.data.objects.new('HEADLINER • midsurface only',me);activecoll.objects.link(panel);panel.parent=tool;panel.location=(0,0,-.62);me.materials.append(cream)
for f in me.polygons:f.use_smooth=True
sub=panel.modifiers.new('Catmull–Clark | open shell midsurface','SUBSURF');sub.levels=2;sub.render_levels=2;sub.boundary_smooth='PRESERVE_CORNERS'
panel['representation']='Single midsurface. No Solidify, caps, back face, or modeled thickness.'
panel['physical_thickness_m']=.003;panel['open_apertures']=len(APERTURES);panel['design']='Generic passenger-car headliner; not an OEM replica.'
panel['formed_details']='Sunroof rebate, visor trays, console/lamp seats, four paired-clip handle pockets, perimeter return'
# Four contact cups, axis aligned to analytic surface normal.
activecoll=toolcoll
cups=[]
for k,(x,y) in enumerate([(-.56,-.65),(.56,-.65),(-.56,.65),(.56,.65)],1):
 z=-.62+height(x,y)
 eps=.0001;hx=(height(x+eps,y)-height(x-eps,y))/(2*eps);hy=(height(x,y+eps)-height(x,y-eps))/(2*eps)
 cup=empty(f'SUCTION {k:02d} • contact',(x,y,z),tool);cup.rotation_mode='QUATERNION';cup.rotation_quaternion=Vector((-hx,-hy,1)).to_track_quat('Z','Y');cups.append(cup)
 # Open bellows profile: open contact rim and tapered stem.
 profile_mm=[(50,0),(51,3),(38,13),(45,22),(32,31),(39,40),(22,51),(17,61)]
 profile=[(r*.001/DESIGN_SCALE,z*.001/DESIGN_SCALE) for r,z in profile_mm]
 vv=[(r*math.cos(a*math.tau/48),r*math.sin(a*math.tau/48),zz) for r,zz in profile for a in range(48)]
 ff=[]
 for j in range(len(profile)-1):
  for a in range(48):ff.append((j*48+a,j*48+(a+1)%48,(j+1)*48+(a+1)%48,(j+1)*48+a))
 mesh=bpy.data.meshes.new('Bellows elastomer');mesh.from_pydata(vv,[],ff);mesh.update();o=bpy.data.objects.new(f'Cup {k} • open bellows',mesh);activecoll.objects.link(o);o.parent=cup;mesh.materials.append(teal)
 for f in mesh.polygons:f.use_smooth=True
 # Plunger and mounting collar are fitted after projection to the smoothed shell.
 curve(f'Vacuum hose {k}',[(x,y,-.29),(x*.85,y*.80,-.10),(x*.35,y*.3,-.10),(0,y*.12,-.12)],HOSE_OUTSIDE_DIAMETER_M/(2*DESIGN_SCALE),teal,tool)
 cup['attachment']='Fixed contact on shell throughout transfer; no slip.'
# Project each cup center and axis onto the actual subdivided midsurface.
from mathutils.bvhtree import BVHTree
bpy.context.view_layer.update()
deps=bpy.context.evaluated_depsgraph_get();ev=panel.evaluated_get(deps);mesh=ev.to_mesh()
tree=BVHTree.FromPolygons([v.co for v in mesh.vertices],[list(p.vertices) for p in mesh.polygons])
for cup in cups:
 local=Vector(cup.location)-panel.location
 nearest,normal,idx,dist=tree.find_nearest(local)
 cup.location=nearest+panel.location
 cup.rotation_quaternion=normal.to_track_quat('Z','Y')
 # The stem reaches the frame's lower face; its axis follows the cup normal.
 stem_start=.061/DESIGN_SCALE
 stem_end=(-.27-cup.location.z)/normal.z
 cyl(cup.name+' • adjustable plunger',(0,0,(stem_start+stem_end)/2),.008/DESIGN_SCALE,stem_end-stem_start,steel,cup)
 cyl(cup.name+' • frame collar',(0,0,stem_end),.022/DESIGN_SCALE,.045/DESIGN_SCALE,dark,cup)
 cyl(cup.name+' • stem fitting',(0,0,stem_start),.019/DESIGN_SCALE,.012/DESIGN_SCALE,steel,cup)
 # Lip follows the local molded surface; the rest of the bellows remains rigid.
 bellows=next(o for o in cup.children if 'open bellows' in o.name)
 rotation=cup.rotation_quaternion.to_matrix()
 for v in bellows.data.vertices[:48]:
  local_surface=rotation@v.co+cup.location-panel.location
  hit=tree.ray_cast(local_surface+Vector((0,0,.08)),Vector((0,0,-1)))
  assert hit[0] is not None
  v.co=rotation.inverted()@(hit[0]+panel.location-cup.location)
# Refit the receiving pads to the actual refined surface, including its normals.
for pad in [o for o in stagecoll.objects if o.name.startswith('Soft nest pad')]:
 station_x=-2.35 if pad.location.x<0 else 2.35
 local=Vector((pad.location.x-station_x,pad.location.y+.35,.5))
 hit=tree.ray_cast(local,Vector((0,0,-1)))
 assert hit[0] is not None
 contact,normal=hit[0],hit[1]
 pad.rotation_mode='QUATERNION';pad.rotation_quaternion=normal.to_track_quat('Z','Y')
 pad.location=Vector((station_x,-.35,1.15))+contact-normal*(.035/2)
 # Refit the vertical post beneath the tilted pad, without object scaling.
 post=min((o for o in stagecoll.objects if o.name.startswith('Locating pedestal')),key=lambda o:(o.location.xy-pad.location.xy).length)
 top=pad.location.z-(.035/2+normal.x*(post.location.x-pad.location.x)+normal.y*(post.location.y-pad.location.y))/normal.z
 bottom=.94;depth=top-bottom
 for vertex in post.data.vertices:vertex.co.z*=depth/.224
 post.location.z=(top+bottom)/2
# Geometry tree was obtained from the current evaluated mesh.
ev.to_mesh_clear()
# Motion: paired transfers, with a short station dwell at each side.
scene.frame_start=1;scene.frame_end=192;scene.render.fps=24
records=[]
def smooth(t):return t*t*t*(10+t*(-15+6*t))
def path(f):
 phase=((f-1)%192)/96
 reverse=phase>=1;t=phase%1
 s=smooth(max(0,min(1,(t-.12)/.76)))
 if reverse:s=1-s
 x=-2.35+4.7*s;z=1.15+1.55*math.sin(math.pi*s)
 return x,-.35,z
for f in range(1,194):
 x,y,z=path(f);dx=x;dy=y-1.65;r=math.hypot(dx,dy);dz=z+.62-1.15;d=math.hypot(r,dz)
 assert abs(L1-L2)<d<L1+L2
 a=math.atan2(dy,dx);theta=math.atan2(dz,r)+math.acos((L1*L1+d*d-L2*L2)/(2*L1*d))
 e_r=L1*math.cos(theta);e_z=L1*math.sin(theta);phi=math.atan2(dz-e_z,r-e_r)
 yaw.rotation_euler[2]=a;shoulder.rotation_euler[1]=-theta;elbow.rotation_euler[1]=theta-phi;wrist.rotation_euler[1]=phi;tool.rotation_euler[2]=-a
 for o in [yaw,shoulder,elbow,wrist,tool]:o.keyframe_insert('rotation_euler',frame=f,group='Baked analytical IK')
 records.append({'frame':f,'headliner_center':[x,y,z],'shoulder_angle':theta,'elbow_absolute_angle':phi})
for o in [yaw,shoulder,elbow,wrist,tool]:
 for fc in o.animation_data.action.fcurves:
  for k in fc.keyframe_points:k.interpolation='LINEAR'
for frame,name in [(1,'A • grip / hold'),(25,'LIFT'),(49,'OVER • clearance'),(85,'B • hold'),(121,'RETURN LIFT'),(145,'OVER'),(181,'A • hold')]:scene.timeline_markers.new(name,frame=frame)
scene['animation_notes']='Four fixed suction contacts. Lift and transfer A to B, then return B to A for a continuous loop. Rigid kinematics, no deformation or release.'
# Studio cameras and light.
activecoll=stagecoll
def camera(name,loc,target,ortho):
 bpy.ops.object.camera_add(location=loc);o=move(bpy.context.object);o.name=name;o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler();o.data.type='ORTHO';o.data.ortho_scale=ortho;return o
cam=camera('CAM • transfer overview',(6.5,-10.5,7.3),(0,0,1.30),10.3);scene.camera=cam
for name,loc,power,size,color in [('Key',(-3,-4,8),2200,5.0,(.82,.91,1)),('Rim',(3,4,7),2800,4,(1,.69,.42)),('Fill',(5,-1,5),1400,4,(.62,.83,1))]:
 bpy.ops.object.light_add(type='AREA',location=loc);o=move(bpy.context.object);o.name=name;o.data.energy=power;o.data.shape='DISK';o.data.size=size;o.data.color=color;o.rotation_euler=(Vector((0,0,1))-o.location).to_track_quat('-Z','Y').to_euler()
scene.world.color=(.16,.16,.16)
scene.render.engine='CYCLES';scene.cycles.samples=32;scene.cycles.use_denoising=True;scene.cycles.denoiser='OPTIX'
prefs=bpy.context.preferences.addons['cycles'].preferences
try:
 prefs.compute_device_type='OPTIX';prefs.get_devices()
 for dev in prefs.devices:dev.use=dev.type=='OPTIX'
 scene.cycles.device='GPU' if any(d.type=='OPTIX' for d in prefs.devices) else 'CPU'
except Exception:scene.cycles.device='CPU'
scene.render.resolution_x=1200;scene.render.resolution_y=900;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.view_settings.view_transform='AgX';scene.view_settings.exposure=-.65
scene.render.film_transparent=False
scene.frame_set(49)
# Resize the first-pass cell to final physical dimensions, baking every length.
# This changes geometry, NOT the unit conversion (one Blender unit stays one meter).
from mathutils import Matrix
scaled_data=set()
for obj in scene.objects:
 obj.location*=DESIGN_SCALE
 data=obj.data
 if obj.type=='MESH' and data not in scaled_data:
  data.transform(Matrix.Scale(DESIGN_SCALE,4));scaled_data.add(data)
 elif obj.type=='CURVE' and data not in scaled_data:
  for spline in data.splines:
   for pt in spline.bezier_points:
    pt.co*=DESIGN_SCALE;pt.handle_left*=DESIGN_SCALE;pt.handle_right*=DESIGN_SCALE
   for pt in spline.points:
    pt.co.xyz*=DESIGN_SCALE
  data.bevel_depth*=DESIGN_SCALE;scaled_data.add(data)
 elif obj.type=='FONT' and data not in scaled_data:
  data.size*=DESIGN_SCALE;data.extrude*=DESIGN_SCALE;scaled_data.add(data)
 elif obj.type=='CAMERA':data.ortho_scale*=DESIGN_SCALE
 elif obj.type=='LIGHT':data.energy*=DESIGN_SCALE**2;data.size*=DESIGN_SCALE
 for mod in obj.modifiers:
  if mod.type=='BEVEL':mod.width*=DESIGN_SCALE
for record in records:record['headliner_center']=[v*DESIGN_SCALE for v in record['headliner_center']]
L1*=DESIGN_SCALE;L2*=DESIGN_SCALE
scene['meters_per_blender_unit']=1.0
scene['design_revision']='Pass 2: physical dimensions baked; nominal 100 mm suction cups; refined forming details'
scene.frame_set(49);bpy.context.view_layer.update()
# Default viewport opens in the presentation camera.
for screen in bpy.data.screens:
 for area in screen.areas:
  if area.type=='VIEW_3D':
   area.spaces.active.region_3d.view_perspective='CAMERA';area.spaces.active.overlay.show_extras=False
bpy.ops.object.select_all(action='DESELECT');panel.select_set(True);bpy.context.view_layer.objects.active=panel
# Topological validation: one connected quad surface, exterior + eleven aperture loops.
counts=Counter(tuple(sorted((face[i],face[(i+1)%4]))) for face in faces for i in range(4));boundary=[e for e,n in counts.items() if n==1];adj=defaultdict(list)
for a,b in boundary:adj[a].append(b);adj[b].append(a)
assert all(len(n)==2 for n in adj.values())
seen=set();loops=[]
for v in adj:
 if v in seen:continue
 stack=[v];loop=[]
 while stack:
  q=stack.pop()
  if q in seen:continue
  seen.add(q);loop.append(q);stack.extend(adj[q])
 loops.append(loop)
assert len(loops)==len(APERTURES)+1 and all(len(f)==4 for f in faces)
report={'vertices':len(verts),'quads':len(faces),'edges':len(counts),'boundary_loops':len(loops),'apertures':len(APERTURES),'nonmanifold_edges_other_than_boundary':sum(n>2 for n in counts.values()),'euler_characteristic':len(verts)-len(counts)+len(faces),'suction_contacts':4,'animation_frames':192,'fps':24,'rigid_link_lengths_m':[L1,L2],'max_headliner_origin_z_m':max(r['headliner_center'][2] for r in records),'shell_thickness_geometry':False,'subdivision_levels':2}
# Validate hierarchy end positions and contact distance to evaluated midsurface using BVH.
from mathutils.bvhtree import BVHTree
errors=[];contact_errors=[];rigid_errors=[]
for f in range(1,193):
 scene.frame_set(f);bpy.context.view_layer.update();want=Vector(records[f-1]['headliner_center']);errors.append((panel.matrix_world.translation-want).length)
 rigid_errors.extend([abs((elbow.matrix_world.translation-shoulder.matrix_world.translation).length-L1),abs((wrist.matrix_world.translation-elbow.matrix_world.translation).length-L2)])
 if f==49:
  deps=bpy.context.evaluated_depsgraph_get();ev=panel.evaluated_get(deps);mesh=ev.to_mesh();tree=BVHTree.FromPolygons([v.co for v in mesh.vertices],[list(p.vertices) for p in mesh.polygons])
  for cup in cups:
   point=panel.matrix_world.inverted()@cup.matrix_world.translation;nearest=tree.find_nearest(point);contact_errors.append(nearest[3])
  ev.to_mesh_clear()
report['max_tool_position_error_m']=max(errors);report['max_rigid_link_length_error_m']=max(rigid_errors);report['suction_center_surface_errors_m']=contact_errors
assert max(errors)<1e-5 and max(rigid_errors)<1e-5 and max(contact_errors)<.002
(ROOT/'validation/geometry.json').write_text(json.dumps(report,indent=2)+'\n')
(ROOT/'validation/motion.json').write_text(json.dumps(records,indent=2)+'\n')
scene.frame_set(49)
from contact_configurations import install
install()
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'scene/headliner_transfer.blend'))
print('VALIDATION',json.dumps(report))
if '--preview' in sys.argv:
 scene.render.filepath=str(ROOT/'build/preview.png');bpy.ops.render.render(write_still=True)
