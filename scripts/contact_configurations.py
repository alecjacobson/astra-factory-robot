"""Two/four suction configurations using native Blender scene/layer exclusions.

All geometry and baked motion are shared. Only complete suction assemblies and
their hoses are excluded in the two-contact scene; no script runs during playback.
"""
import bpy
MAIN_SCENE='01 • HEADLINER TRANSFER'
TWO_SCENE='06 • TWO-CONTACT TRANSFER'
TOOL_COLLECTION='02 • Vacuum tooling — configurable contacts'
OPTIONAL_COLLECTION='Optional contacts 02 + 03 • four-contact mode only'
LAYOUTS={4:(1,2,3,4),2:(1,4)}

def layer_collection(root,name):
    if root.name==name:return root
    for child in root.children:
        result=layer_collection(child,name)
        if result:return result
    return None

def configure(scene,count):
    if count not in LAYOUTS:raise ValueError('Contact count must be 2 or 4')
    optional=layer_collection(scene.view_layers[0].layer_collection,OPTIONAL_COLLECTION)
    if optional is None:raise ValueError('Contact collections are not installed')
    optional.exclude=count==2
    scene['suction_contact_count']=count
    scene['active_suction_contact_ids']=list(LAYOUTS[count])
    scene['contact_configuration']='Four corner contacts' if count==4 else 'Opposing diagonal contacts 01 + 04'
    scene['stability_model']='Kinematic prescribed motion; no suction, friction, shell mechanics, or stability simulation.'
    scene['animation_notes']=f'{count} fixed suction contacts. A to B and return; identical prescribed path in both layouts.'
    scene.view_layers[0].name=f'{count} suction contacts'

def install():
    main=bpy.data.scenes[MAIN_SCENE]
    tool_collection=bpy.data.collections.get(TOOL_COLLECTION) or bpy.data.collections.get('02 • Vacuum tooling — four contacts')
    if tool_collection is None:raise ValueError('Vacuum tooling collection is missing')
    tool_collection.name=TOOL_COLLECTION
    optional=bpy.data.collections.get(OPTIONAL_COLLECTION)
    if optional is None:
        optional=bpy.data.collections.new(OPTIONAL_COLLECTION)
        tool_collection.children.link(optional)
    for index in [2,3]:
        root=bpy.data.objects[f'SUCTION {index:02d} • contact']
        objects=[root,*root.children_recursive,bpy.data.objects[f'Vacuum hose {index}']]
        for obj in objects:
            if obj.name not in optional.objects:optional.objects.link(obj)
            if obj.name in tool_collection.objects:tool_collection.objects.unlink(obj)
    configure(main,4)
    two=bpy.data.scenes.get(TWO_SCENE)
    if two is None:
        two=main.copy();two.name=TWO_SCENE
    two.use_fake_user=True
    bpy.context.window.scene=two
    configure(two,2)
    two.camera=main.camera
    two.frame_start=main.frame_start;two.frame_end=main.frame_end;two.frame_set(49)
    note=bpy.data.texts.get('START HERE • contact configurations') or bpy.data.texts.new('START HERE • contact configurations')
    note.clear();note.write('''SUCTION CONTACT OPTIONS

Choose a scene from the Scene dropdown in Blender's top bar:
  01 • HEADLINER TRANSFER     four corner suction contacts
  06 • TWO-CONTACT TRANSFER  diagonal contacts 01 and 04 only

The robot, shell, tooling frame, camera, and baked motion are shared.
The two-contact scene excludes the entire optional 02/03 assembly collection:
cups, stems, mounting collars, fittings, contact markers, and vacuum hoses.
The four support pads on each stationary receiving fixture are unchanged.

Press Space to play. F12 renders the selected scene. No add-ons, drivers,
handlers, or automatic Python execution are needed for switching or playback.

This is a prescribed kinematic comparison, not a prediction of stability.
The headliner remains parented to the tool in both scenes. The reduced-contact
case is prepared for later mechanical analysis; no wobble or failure is implied.
''')
    bpy.context.window.scene=main;main.frame_set(49)
    return main,two

if __name__=='__main__':
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    install()
    bpy.ops.wm.save_as_mainfile(filepath=str(root/'scene/headliner_transfer.blend'))
