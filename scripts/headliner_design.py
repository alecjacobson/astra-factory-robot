"""Headliner forming layout. Coordinates are design meters before the 0.85 resize.

The final scene bakes DESIGN_SCALE into mesh data and local translations, leaving
unit object scales and one Blender unit = one physical meter. This resizes the
first-pass cell consistently, rather than changing Blender's unit conversion.
"""
import math
DESIGN_SCALE = 0.85
CUP_DIAMETER_M = 0.100
HOSE_OUTSIDE_DIAMETER_M = 0.010
# x min/max, y min/max, corner radius, feature name
APERTURES = [
    (-.40,.40,-.35,.45,.060,'Sunroof'),
    (-.15,.15,.735,.925,.045,'Front overhead console'),
    (-.105,.105,-.865,-.725,.035,'Rear reading lamp'),
]
# Paired clip-access slots in four grab-handle mounting recesses.
for side in [-1,1]:
    for center_y in [-.40,.40]:
        for offset in [-.065,.065]:
            cx=side*.685; cy=center_y+offset
            APERTURES.append((cx-.015,cx+.015,cy-.020,cy+.020,.009,'Grab-handle clip access'))

def taper(y):
    return 1-.065*(y/1.1)-.075*(abs(y)/1.1)**10

def rounded_distance(x,y,cx,cy,hx,hy,r):
    qx=abs(x-cx)-(hx-r);qy=abs(y-cy)-(hy-r)
    return math.hypot(max(qx,0),max(qy,0))+min(max(qx,qy),0)-r

def height(x,y):
    # Convert back to the forming grid so features follow the tapered panel.
    u=x/taper(y)
    z=.085*(1-(x/.8)**2)+.030*(1-(y/1.1)**2)
    # Turned-down perimeter and shallow side-rail return, all still a midsurface.
    edge=max(0,-rounded_distance(u,y,0,0,.8,1.1,.12))
    z-=.060*math.exp(-(edge/.055)**2)
    z+=.017*math.exp(-((edge-.095)/.040)**2)
    z-=.020*math.exp(-((1.1-abs(y))/.050)**2)
    # Sunroof flange: narrow raised shoulder followed by a recessed outer reveal.
    sd=rounded_distance(u,y,0,.05,.40,.40,.060)
    z+=.021*math.exp(-((sd-.027)/.021)**2)
    z-=.006*math.exp(-((sd-.075)/.027)**2)
    # Console and lamp seats with a formed shoulder around each open aperture.
    for cx,cy,hx,hy,r in [(0,.83,.195,.14,.055),(0,-.795,.155,.12,.045)]:
        d=rounded_distance(u,y,cx,cy,hx,hy,r)
        z-=.026/(1+math.exp(max(-60,min(60,d/.012))))
        z+=.006*math.exp(-(d/.019)**2)
    # Shallow, rounded visor trays forward of the sunroof.
    for cx in [-.415,.415]:
        d=rounded_distance(u,y,cx,.88,.205,.105,.045)
        z-=.024/(1+math.exp(max(-60,min(60,d/.012))))
        z+=.005*math.exp(-(d/.016)**2)
    # Local molded pockets for each pair of grab-handle clip slots.
    for cx in [-.685,.685]:
        for cy in [-.4,.4]:
            d=rounded_distance(u,y,cx,cy,.052,.123,.035)
            z-=.022/(1+math.exp(max(-60,min(60,d/.009))))
    # A subtle transverse formed stiffening contour behind the sunroof.
    z+=.009*math.exp(-((y+.55)/.027)**2)*math.exp(-(u/.55)**8)
    return z

def build_mesh():
    # A 25 mm forming grid, with feature-aligned edge/support samples.
    xs={round(-.8+i*.025,6) for i in range(65)}
    ys={round(-1.1+j*.025,6) for j in range(89)}
    for xmin,xmax,ymin,ymax,r,name in APERTURES:
        for a in [xmin,xmax]:
            xs.update([round(a,6),round(a-.009,6),round(a+.009,6)])
        for a in [ymin,ymax]:
            ys.update([round(a,6),round(a-.009,6),round(a+.009,6)])
    # Avoid sliver intervals when a support sample almost coincides with the base grid.
    def merge(values):
        mandatory={round(v,6) for h in APERTURES for v in (h[:2] if values is xs else h[2:4])}
        ordered=sorted(values);out=[]
        for v in ordered:
            if out and v-out[-1]<.003:
                if v in mandatory:out[-1]=v
            else:out.append(v)
        return out
    xx=merge(xs);yy=merge(ys);nv=len(xx);verts=[];faces=[]
    for y in yy:
        for u in xx:
            ru,ry=u,y
            # Round the boundary AND its surrounding support rows together.
            # Smoothly fading the corner warp prevents pinched subdivision shoulders.
            for xmin,xmax,ymin,ymax,r,name in APERTURES:
                cx=(xmin+xmax)/2;cy=(ymin+ymax)/2;hx=(xmax-xmin)/2;hy=(ymax-ymin)/2
                qx=abs(u-cx);qy=abs(y-cy);distance=max(qx-hx,qy-hy)
                vx=qx-(hx-r);vy=qy-(hy-r);band=2*r
                if vx>0 and vy>0 and -.0000001<=distance<band:
                    d=max(0,distance);t=d/band
                    fade=1-t*t*t*(10+t*(-15+6*t))
                    factor=1+((r+d)/math.hypot(vx,vy)-1)*fade
                    ru=cx+math.copysign(hx-r+vx*factor,u-cx)
                    ry=cy+math.copysign(hy-r+vy*factor,y-cy)
            # Smoothly map the outer corner patches onto a 120 mm forming radius.
            q=max(abs(ru)/.8,abs(ry)/1.1)
            if q>0:
                bu,bv=ru/q,ry/q
                if rounded_distance(bu,bv,0,0,.8,1.1,.12)>0:
                    lo,hi=0.,1.
                    for _ in range(24):
                        mid=(lo+hi)/2
                        if rounded_distance(mid*bu,mid*bv,0,0,.8,1.1,.12)>0:hi=mid
                        else:lo=mid
                    factor=1-(1-(lo+hi)/2)*q**12
                    ru*=factor;ry*=factor
            x=ru*taper(ry);verts.append((x,ry,height(x,ry)))
    for j in range(len(yy)-1):
        for i in range(len(xx)-1):
            cx=(xx[i]+xx[i+1])/2;cy=(yy[j]+yy[j+1])/2
            if any(xmin<cx<xmax and ymin<cy<ymax for xmin,xmax,ymin,ymax,r,name in APERTURES):continue
            a=j*nv+i;faces.append((a,a+1,a+nv+1,a+nv))
    used=sorted({v for f in faces for v in f});mapping={v:i for i,v in enumerate(used)}
    return [verts[v] for v in used],[tuple(mapping[v] for v in f) for f in faces]
