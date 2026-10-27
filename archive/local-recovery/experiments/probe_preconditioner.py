import numpy as np
import warp as wp
from simulation.model import load_headliner
from simulation.dynamics import Integrator
from simulation import bac_hessian
@wp.kernel
def gn(jac:wp.array3d[wp.float64],h:wp.array3d[wp.float64]):
 f,i,j=wp.tid();v=wp.float64(0)
 for k in range(8):v+=jac[f,k,i]*jac[f,k,j]
 h[f,i,j]=v
wp.init();wp.set_mempool_release_threshold('cuda:0',0)
s,_=load_headliner(2);h=s.stiffness();a=Integrator(s,h,dt=1/960,max_steps=1)
a.x.assign(wp.array(np.load('build/two-960/full-debug.npz')['state'],dtype=wp.float64));s.evaluate(a.x)
wp.launch(bac_hessian.evaluate,a.hessian.shape,[a.x,s.ids,s.signs,s.interior,s.rest,s.young,s.nu,s.thickness,s.fixed,a.hessian])
for name in ['consistent','gauss-newton']:
 if name=='gauss-newton':wp.launch(gn,a.hessian.shape,[s.jac,a.hessian])
 a.refit.counts.zero_()
 with wp.ScopedCapture() as cap:a.refit.update(a.hessian)
 wp.capture_launch(cap.graph);wp.synchronize()
 print(name,a.refit.counts.numpy().tolist(),a.refit.plan.bad.numpy().tolist(),flush=True)
