import time,numpy as np
import warp as wp
from warp.optim import linear
from simulation.model import load_headliner
from simulation.reduction import build_basis
from simulation.dynamics import Integrator,project,lift,dense_mv,combine
from simulation.bac import I21
@wp.func
def pos(x:wp.array[wp.float64],i:int):return wp.vec3d(x[3*i],x[3*i+1],x[3*i+2])
@wp.func
def frame(a:wp.vec3d,b:wp.vec3d):
 n=wp.normalize(wp.cross(a,b));return wp.mat33d(a[0],b[0],n[0],a[1],b[1],n[1],a[2],b[2],n[2])
@wp.kernel
def accumulate(x:wp.array[wp.float64],x0:wp.array[wp.float64],ids:wp.array[I21],sums:wp.array[wp.mat33d]):
 f=wp.tid();a=ids[f][0]//3;b=ids[f][3]//3;c=ids[f][6]//3
 r1=pos(x0,b)-pos(x0,a);r2=pos(x0,c)-pos(x0,a)
 F=frame(pos(x,b)-pos(x,a),pos(x,c)-pos(x,a))*wp.inverse(frame(r1,r2))
 U,s,V=wp.svd3(F);R=U*wp.transpose(V)*wp.length(wp.cross(r1,r2))
 wp.atomic_add(sums,a,R);wp.atomic_add(sums,b,R);wp.atomic_add(sums,c,R)
@wp.kernel
def polar(sums:wp.array[wp.mat33d],rot:wp.array[wp.mat33d]):
 i=wp.tid();U,s,V=wp.svd3(sums[i]);R=U*wp.transpose(V)
 if wp.determinant(R)<wp.float64(0):
  U[0,2]=-U[0,2];U[1,2]=-U[1,2];U[2,2]=-U[2,2];R=U*wp.transpose(V)
 rot[i]=R
@wp.kernel
def rotate(x:wp.array[wp.float64],R:wp.array[wp.mat33d],out:wp.array[wp.float64],nv:int,backward:int):
 i=wp.tid();v=x[i]
 if i<3*nv:
  m=R[i//3]
  if backward!=0:m=wp.transpose(m)
  a=m*pos(x,i//3);v=a[i%3]
 out[i]=v
@wp.kernel
def load(m:wp.array[wp.float64],fixed:wp.array[int],b:wp.array[wp.float64],nv:int):
 i=wp.tid();b[i]=wp.float64(0)
 if i<3*nv and i%3==2 and fixed[i]==0:b[i]=-wp.float64(9.81)*m[i]
wp.init();wp.set_mempool_release_threshold('cuda:0',0)
s,_=load_headliner(2);b,h,_=build_basis(s,40);sim=Integrator(s,h,pre_basis=b,dt=1/960,max_steps=1)
sim.x.assign(wp.array(np.load('build/comparison-2/full-failed.npz')['state'],dtype=wp.float64));s.evaluate(sim.x)
with wp.ScopedCapture() as cap:sim.refresh_hessian()
wp.capture_launch(cap.graph);wp.synchronize()
wp.launch(load,s.nd,[s.mass,s.fixed,sim.rhs,s.nv]);R=wp.zeros(s.nv,dtype=wp.mat33d);sm=wp.zeros_like(R)
wp.launch(accumulate,s.nf,[sim.x,s.x0,s.ids,sm]);wp.launch(polar,s.nv,[sm,R]);nr=b.shape[0];n=s.nd
rotated=wp.zeros(n,dtype=wp.float64);full=wp.zeros_like(rotated);rotfull=wp.zeros_like(full);q=wp.zeros(nr,dtype=wp.float64);sol=wp.zeros_like(q)
def prec(x,y,z,alpha,beta):
 sim.refit.inverse.matvec(x,y,z,alpha,beta)
 wp.launch(rotate,n,[x,R,rotated,s.nv,1]);q.zero_();wp.launch_tiled(project,dim=(nr,(n+255)//256),inputs=[b,rotated,q],block_dim=128)
 wp.launch(dense_mv,nr,[sim.inv,q,q,sol,1.,0.]);wp.launch(lift,n,[b,sol,full]);wp.launch(rotate,n,[full,R,rotfull,s.nv,0]);wp.launch(combine,n,[rotfull,z,z,alpha,1.])
for label,pre in [('rest coarse',sim.pre),('rotated coarse',linear.LinearOperator((n,n),wp.float64,wp.get_device(),prec))]:
 sim.cg.m=pre
 with wp.ScopedCapture() as cap:sim.cg()
 wp.capture_launch(cap.graph);wp.synchronize();t=time.perf_counter();wp.capture_launch(cap.graph);wp.synchronize()
 print(label,'iterations',sim.cg.it.numpy().tolist(),'residual',sim.cg.rr.numpy().tolist(),'negative',sim.cg.negative.numpy().tolist(),'seconds',time.perf_counter()-t,flush=True)
