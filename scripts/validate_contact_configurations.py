"""Check actual scene exclusions, contact geometry, and identical 192-frame motion."""
import bpy,json,sys
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from contact_configurations import MAIN_SCENE,TWO_SCENE,LAYOUTS,OPTIONAL_COLLECTION,layer_collection
panel=bpy.data.objects['HEADLINER • midsurface only']
cups={i:bpy.data.objects[f'SUCTION {i:02d} • contact'] for i in range(1,5)}
expected=json.loads((ROOT/'validation/motion.json').read_text())
main=bpy.data.scenes[MAIN_SCENE];bpy.context.window.scene=main;main.frame_set(49);bpy.context.view_layer.update()
ev=panel.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
surface=BVHTree.FromPolygons([v.co.copy() for v in mesh.vertices],[tuple(f.vertices) for f in mesh.polygons]);ev.to_mesh_clear()
report={'comparison':'Identical prescribed motion; visibility and geometric contacts verified, no mechanical stability simulation.','two_contact_pair':[1,4],'configurations':{}}
for count,name in [(4,MAIN_SCENE),(2,TWO_SCENE)]:
 scene=bpy.data.scenes[name];bpy.context.window.scene=scene
 assert scene.unit_settings.system=='METRIC' and scene.unit_settings.scale_length==1
 assert scene.camera is main.camera
 assert bool(layer_collection(scene.view_layers[0].layer_collection,OPTIONAL_COLLECTION).exclude)==(count==2)
 assert list(scene['active_suction_contact_ids'])==list(LAYOUTS[count])
 max_center=0;max_lip=0;max_path=0;poses=[]
 for frame in range(1,193):
  scene.frame_set(frame);bpy.context.view_layer.update();layer=scene.view_layers[0];deps=bpy.context.evaluated_depsgraph_get()
  visible=[i for i,cup in cups.items() if cup.visible_get(view_layer=layer)]
  assert visible==list(LAYOUTS[count]),(name,frame,visible)
  panel_world=panel.evaluated_get(deps).matrix_world
  inverse=panel_world.inverted()
  max_path=max(max_path,(panel_world.translation-Vector(expected[frame-1]['headliner_center'])).length)
  poses.append([list(row) for row in panel_world])
  for i,cup in cups.items():
   included=i in LAYOUTS[count]
   for obj in [cup,*cup.children_recursive,bpy.data.objects[f'Vacuum hose {i}']]:
    assert obj.visible_get(view_layer=layer)==included,(name,frame,obj.name)
    if included:assert not obj.hide_render
   if not included:continue
   max_center=max(max_center,surface.find_nearest(inverse@cup.evaluated_get(deps).matrix_world.translation)[3])
   bellows=next(o for o in cup.children if 'open bellows' in o.name)
   matrix=inverse@bellows.evaluated_get(deps).matrix_world
   for v in bellows.data.vertices[:48]:max_lip=max(max_lip,surface.find_nearest(matrix@v.co)[3])
 assert max_center<1e-5 and max_lip<2e-5 and max_path<1e-5,(count,max_center,max_lip,max_path)
 if count==4:reference_poses=poses
 else:assert max(abs(v-w) for a,b in zip(poses,reference_poses) for row_a,row_b in zip(a,b) for v,w in zip(row_a,row_b))<1e-6
 scene.frame_set(49);bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get();inverse=panel.evaluated_get(deps).matrix_world.inverted()
 report['configurations'][str(count)]={'scene':name,'active_contact_ids':list(LAYOUTS[count]),'visible_suction_assemblies':count,'visible_vacuum_hoses':count,'frames_checked':192,'maximum_contact_center_error_m':max_center,'maximum_contact_lip_error_m':max_lip,'maximum_path_error_m':max_path,'contact_centers_in_headliner_coordinates_m':{str(i):list(inverse@cups[i].evaluated_get(deps).matrix_world.translation) for i in LAYOUTS[count]},'nominal_contact_diameter_m':.1}
report['identical_headliner_transform_all_frames']=True
(ROOT/'validation/contact-configurations.json').write_text(json.dumps(report,indent=2)+'\n')
bpy.context.window.scene=main;main.frame_set(49)
print(json.dumps(report,indent=2))
