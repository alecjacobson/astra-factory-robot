import time,numpy as np
import warp as wp
from simulation.model import load_headliner
from simulation.dynamics import Integrator
wp.init();wp.set_mempool_release_threshold('cuda:0',0)
s,_=load_headliner(2);h=s.stiffness();a=Integrator(s,h,dt=1/960,max_steps=1)
a.x.assign(wp.array(np.load('build/comparison-2/full-failed.npz')['state'],dtype=wp.float64));s.evaluate(a.x)
with wp.ScopedCapture() as cap:a.refresh_hessian()
wp.capture_launch(cap.graph);wp.synchronize()
rhs=s.mass.numpy();rhs[:]=rhs*(-9.81);rhs[3*s.nv:]=0;rhs[np.arange(s.nd)%3!=2]=0;rhs[np.array(s.fixed_host).astype(bool)]=0;a.rhs.assign(wp.array(rhs,dtype=wp.float64))
with wp.ScopedCapture() as cap:a.cg()
wp.capture_launch(cap.graph);wp.synchronize();t=time.perf_counter();wp.capture_launch(cap.graph);wp.synchronize()
print('FSAI only',a.cg.it.numpy().tolist(),a.cg.rr.numpy().tolist(),a.cg.negative.numpy().tolist(),time.perf_counter()-t,flush=True)
