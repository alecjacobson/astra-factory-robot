"""Label rendered views and encode a 12-fps GIF (requires Pillow and gifski)."""
import argparse, math, subprocess, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--gifski',default='gifski');p.add_argument('--font-dir',default='/usr/share/fonts/truetype/dejavu');p.add_argument('--contacts',type=int,choices=[2,4],default=4);args=p.parse_args()
FONT=Path(args.font_dir)
geometry=json.loads((ROOT/'validation/geometry.json').read_text())
dimensions=json.loads((ROOT/'validation/dimensions.json').read_text())
motion=json.loads((ROOT/'validation/motion.json').read_text())
def font(size,bold=False):
 return ImageFont.truetype(str(FONT/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')),size)
BG=(16,25,33);WHITE=(234,239,238);MUTED=(150,174,184);TEAL=(83,211,194)
def card(src,title,kicker,footer,technical=False):
 im=Image.open(src).convert('RGB');w,h=im.size;scale=w/1200;head=int(120*scale) if technical else 0;foot=int(72*scale)
 out=Image.new('RGB',(w,h+head+foot),BG);out.paste(im,(0,head));d=ImageDraw.Draw(out)
 x=int(34*scale)
 d.text((x,int(22*scale)),kicker,font=font(int(13*scale),True),fill=TEAL)
 d.text((x,int(48*scale)),title,font=font(int(32*scale),True),fill=WHITE)
 d.text((w-int(156*scale),int(30*scale)),'HL / 04',font=font(int(23*scale),True),fill=MUTED)
 d.line((x,int(102*scale),w-x,int(102*scale)),fill=(68,87,98),width=1)
 d.text((x,h+head+int(24*scale)),footer,font=font(int(15*scale)),fill=MUTED)
 return out
media=ROOT/'media';media.mkdir(exist_ok=True)
if args.contacts==4:
 card(ROOT/'build/hero.png','Headliner transfer','AUTOMOTIVE  /  ROBOTIC HANDLING','4 SUCTION CONTACTS     /     RIGID ARM     /     OPEN SUBDIVISION SHELL').save(media/'hero.png',optimize=True)
 views=[('headliner_wireframe','headliner-wireframe','Headliner / control cage','MESH INSPECTION  /  01',f"{geometry['quads']:,} QUADS    /    ORANGE: OPEN BOUNDARIES    /    TEAL: SUCTION FOOTPRINTS"),('subdivision_wireframe','subdivision-wireframe','Headliner / subdivision','MESH INSPECTION  /  02','CATMULL–CLARK    /    EVALUATED QUAD EDGES    /    ELEVEN OPEN APERTURES'),('robot_wireframe','robot-wireframe','Robot / rigid assembly','MESH INSPECTION  /  03','RIGID CASTINGS    /    ROTATING JOINTS    /    FOUR-CUP VACUUM FRAME')]
 for src,dst,title,kicker,footer in views:
  card(ROOT/f'build/{src}.png',title,kicker,footer,True).save(media/f'{dst}.png',optimize=True)
 width,length,depth=dimensions['headliner_evaluated_width_length_depth_m']
 card(ROOT/'build/headliner_detail.png','Headliner / formed midsurface','DETAIL STUDY  /  TRUE METRIC SCALE',f'{width*1000:.0f} × {length*1000:.0f} mm     /     {depth*1000:.0f} mm FORM DEPTH     /     11 OPEN APERTURES',True).save(media/'headliner-detail.png',optimize=True)
else:
 card(ROOT/'build/hero-two-contacts.png','Headliner transfer / 2 contacts','CONTACT STUDY  /  DIAGONAL PAIR 01 + 04','2 SUCTION CONTACTS     /     SAME PRESCRIBED MOTION     /     STABILITY NOT SIMULATED').save(media/'hero-two-contacts.png',optimize=True)
# Rendered forward frames 1,3,...,97. Reverse mapping exactly matches the baked return trajectory.
forward=list(range(1,98,2));sequence=forward[:-1]+forward[:0:-1]
frame_folder=ROOT/('build/frames-2' if args.contacts==2 else 'build/frames')
output_file=media/('headliner-transfer-2-contacts.gif' if args.contacts==2 else 'headliner-transfer.gif')
folder=ROOT/('build/gif-2' if args.contacts==2 else 'build/gif');folder.mkdir(exist_ok=True);files=[]
for i,f in enumerate(sequence):
 t=(f-1)/96;s=max(0,min(1,(t-.12)/.76));s=s*s*s*(10+s*(-15+6*s));z=motion[f-1]['headliner_center'][2]
 direction='A → B' if i<48 else 'B → A'
 motion_t=t if i<48 else 1-t
 phase='STATION HOLD' if motion_t<.12 or motion_t>.88 else ('LIFT / TRANSFER' if motion_t<.5 else 'TRANSFER / LOWER')
 im=card(frame_folder/f'{f:04d}.png','Headliner transfer / 2 contacts' if args.contacts==2 else 'Headliner transfer','2-CONTACT STUDY  /  PRESCRIBED MOTION' if args.contacts==2 else 'AUTOMOTIVE  /  ROBOTIC HANDLING',f'{direction}     |     {phase}     |     HEIGHT {z:.2f} m     |     {args.contacts} CONTACTS')
 d=ImageDraw.Draw(im);w,h=im.size;d.rectangle((0,h-4,w,h),fill=(45,63,72));d.rectangle((0,h-4,int(w*(i+1)/96),h),fill=TEAL)
 path=folder/f'{i:04d}.png';im.save(path);files.append(str(path))
subprocess.run([args.gifski,'--fps','12','--quality','85','--repeat','0','--output',str(output_file),*files],check=True)
im=Image.open(output_file);duration=0
for i in range(im.n_frames):im.seek(i);duration+=im.info.get('duration',0)
print(f'GIF: {im.n_frames} frames, {duration} ms, {im.size}, {output_file.stat().st_size:,} bytes')
