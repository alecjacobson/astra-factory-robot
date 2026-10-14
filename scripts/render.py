"""Render presentation frames and save inspectable topology scenes."""
import bpy, math, sys
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from headliner_design import DESIGN_SCALE, CUP_DIAMETER_M
main=bpy.context.scene
prefs=bpy.context.preferences.addons['cycles'].preferences
try:
 prefs.compute_device_type='OPTIX';prefs.get_devices()
 for device in prefs.devices:device.use=device.type=='OPTIX'
 main.cycles.device='GPU' if any(d.type=='OPTIX' for d in prefs.devices) else 'CPU'
except Exception:main.cycles.device='CPU'
mode=sys.argv[sys.argv.index('--')+1] if '--' in sys.argv else 'animation'
def render(s,path,w=1200,h=900,samples=48):
 bpy.context.window.scene=s;s.render.resolution_x=w;s.render.resolution_y=h;s.render.resolution_percentage=100;s.cycles.samples=samples;s.cycles.denoiser='OPTIX' if s.cycles.device=='GPU' else 'OPENIMAGEDENOISE';s.render.filepath=str(path);bpy.ops.render.render(write_still=True)
if mode=='animation':
 main.render.use_persistent_data=True
 for f in range(1,98,2):
  main.frame_set(f)
  path=ROOT/f'build/frames/{f:04d}.png'
  if not path.exists():render(main,path,960,720,24)
else:
 main.frame_set(49)
 render(main,ROOT/'build/hero.png',1440,1080,64)
 for f in [1,25,73,97]:
  main.frame_set(f);render(main,ROOT/f'build/pose_{f:03d}.png',960,720,32)
 main.frame_set(49)
 def material(name,color,metal=0):
  m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Metallic'].default_value=metal;p.inputs['Roughness'].default_value=.6;return m
 navy=material('Topology | navy',(.025,.075,.105));ivory=material('Topology | ivory',(.58,.70,.72));line=material('Topology | cage',(.018,.036,.045));edge=material('Topology | open edges',(.95,.25,.045));turq=material('Topology | cyan',(.02,.7,.6))
 def scene_copy(name):
  s=bpy.data.scenes.new(name);s.unit_settings.system='METRIC';s.unit_settings.scale_length=1.0;s.unit_settings.length_unit='METERS';s.world=main.world;s.render.engine='CYCLES';s.cycles.device=main.cycles.device;s.cycles.use_denoising=True;s.view_settings.view_transform='AgX';s.view_settings.exposure=-.65
  for o in main.objects:
   if o.type=='LIGHT':s.collection.objects.link(o)
  return s
 def camera(s,loc,target,scale):
  d=bpy.data.cameras.new(s.name);o=bpy.data.objects.new('CAM • '+s.name,d);s.collection.objects.link(o);o.location=Vector(loc)*DESIGN_SCALE;o.rotation_euler=(Vector(target)*DESIGN_SCALE-o.location).to_track_quat('-Z','Y').to_euler();d.type='ORTHO';d.ortho_scale=scale*DESIGN_SCALE;s.camera=o
 def wire(s,source,thickness=.003,material=line):
  o=source.copy();o.data=source.data.copy();o.name='WIREFRAME • '+source.name;s.collection.objects.link(o);o.modifiers.clear();o.data.materials.clear();o.data.materials.append(material)
  for p in o.data.polygons:p.material_index=0
  m=o.modifiers.new('Actual control mesh edges','WIREFRAME');m.thickness=thickness;m.use_replace=True;m.use_boundary=True;m.offset=1
  return o
 def polyline(s,name,points,r,mat):
  d=bpy.data.curves.new(name,'CURVE');d.dimensions='3D';d.bevel_depth=r;d.bevel_resolution=2
  spl=d.splines.new('POLY');spl.points.add(len(points)-1)
  for p,co in zip(spl.points,points):p.co=(*co,1)
  o=bpy.data.objects.new(name,d);s.collection.objects.link(o);d.materials.append(mat);return o
 source=bpy.data.objects['HEADLINER • midsurface only']
 s=scene_copy('02 • Headliner control topology')
 p=source.copy();p.data=source.data.copy();p.parent=None;p.animation_data_clear();p.location=(0,0,0);p.rotation_euler=(0,0,0);p.data.materials.clear();p.data.materials.append(ivory);s.collection.objects.link(p)
 # Use the unsubdivided surface beneath its exact cage; subdivision remains inspectable in main scene.
 p.modifiers.clear()
 wire(s,p,.0012)
 from collections import Counter
 ec=Counter(tuple(sorted((f.vertices[i],f.vertices[(i+1)%len(f.vertices)]))) for f in p.data.polygons for i in range(len(f.vertices)))
 for (a,b),n in ec.items():
  if n==1:polyline(s,'Open boundary', [p.data.vertices[a].co+Vector((0,0,.004)),p.data.vertices[b].co+Vector((0,0,.004))],.0025,edge)
 for k in range(1,5):
  cup=bpy.data.objects[f'SUCTION {k:02d} • contact'];pos=Vector(cup.location)-source.location
  polyline(s,f'Contact footprint {k}',[pos+Vector((CUP_DIAMETER_M/2*math.cos(a*math.tau/64),CUP_DIAMETER_M/2*math.sin(a*math.tau/64),.015)) for a in range(65)],.0025,turq)
 camera(s,(.15,-.65,5),(0,0,0),2.65)
 render(s,ROOT/'build/headliner_wireframe.png',1200,1200,48)
 # Subdivision result: a second inspectable scene with an evaluated level-1 mesh wire overlay.
 s2=scene_copy('03 • Subdivision surface topology')
 q=source.copy();q.data=source.data.copy();q.parent=None;q.location=(0,0,0);q.rotation_euler=(0,0,0);q.animation_data_clear();q.data.materials.clear();q.data.materials.append(ivory);s2.collection.objects.link(q)
 bpy.context.window.scene=s2;bpy.context.view_layer.update()
 q.modifiers[0].levels=1;q.modifiers[0].render_levels=1
 deps=bpy.context.evaluated_depsgraph_get();evaluated=q.evaluated_get(deps);mesh=bpy.data.meshes.new_from_object(evaluated)
 wo=bpy.data.objects.new('Level 1 • evaluated quad edges',mesh);s2.collection.objects.link(wo);wo.data.materials.clear();wo.data.materials.append(line)
 mod=wo.modifiers.new('Subdivision edges','WIREFRAME');mod.thickness=.00065;mod.offset=1;mod.use_boundary=True
 q.modifiers[0].levels=2;q.modifiers[0].render_levels=2
 camera(s2,(2,-3,4.8),(0,0,0),2.85)
 render(s2,ROOT/'build/subdivision_wireframe.png',1200,1100,48)
 # Robot rigid-body mesh inspection, at a less foreshortened three-quarter angle.
 s3=scene_copy('04 • Robot rigid mesh topology')
 for collname in ['01 • Robot — rigid links','02 • Vacuum tooling — four contacts','03 • Headliner — open midsurface']:
  for original in bpy.data.collections[collname].objects:
   if original.type!='MESH':continue
   o=original.copy();o.data=original.data.copy();o.animation_data_clear();o.parent=None;o.matrix_world=original.matrix_world.copy();s3.collection.objects.link(o);o.data.materials.clear();o.data.materials.append(navy if collname.startswith('01') else ivory)
   for poly in o.data.polygons:poly.material_index=0
   if 'HEADLINER' not in original.name:wire(s3,o,.0025,turq)
   else:wire(s3,o,.002,line)
 camera(s3,(5,-7,4.7),(0,.75,1.6),4.6)
 render(s3,ROOT/'build/robot_wireframe.png',1200,1200,48)

 s4=bpy.data.scenes.new('05 • Headliner formed details');s4.unit_settings.system='METRIC';s4.unit_settings.scale_length=1;s4.unit_settings.length_unit='METERS'
 s4.world=main.world;s4.render.engine='CYCLES';s4.cycles.device=main.cycles.device;s4.cycles.use_denoising=True;s4.view_settings.view_transform='AgX';s4.view_settings.exposure=-.4
 detail=source.copy();detail.data=source.data.copy();detail.parent=None;detail.location=(0,0,0);detail.rotation_euler=(0,0,0);detail.animation_data_clear();s4.collection.objects.link(detail)
 for name,loc,power,size,color in [('Grazing key',(-1.5,-1.5,2.2),150,1.2,(1,.87,.70)),('Soft fill',(2,-.5,2),55,2,(.73,.87,1)),('Edge light',(-.5,2,1.3),180,1.5,(1,1,1))]:
  data=bpy.data.lights.new(name,'AREA');light=bpy.data.objects.new(name,data);s4.collection.objects.link(light);light.location=loc;light.rotation_euler=(-light.location).to_track_quat('-Z','Y').to_euler();data.energy=power;data.size=size;data.color=color
 camera(s4,(1.8,-2.5,3.8),(0,0,0),2.9)
 render(s4,ROOT/'build/headliner_detail.png',1400,1300,80)
 bpy.context.window.scene=main;main.frame_set(49);main.render.resolution_x=1200;main.render.resolution_y=900;main.cycles.samples=64
 main.render.filepath='//../build/frames/'
 bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'scene/headliner_transfer.blend'))
