# CPU replica of the XPBD grain contact in sim_matter_grain.glsl (xpbdConstraint +
# the XPBD integration tail) for one grain on a plane. Build-free reference for the
# friction/rolling rule; run: python scripts/test/xpbd_grain_friction_reference.py
# Live 2026-10-06: the old rule (static correction through the 3.5/m rolling inverse
# mass, spin cap afterwards) crept at 2.5 h g sin(theta) on a held slope.
import numpy as np, math, sys
def run(S, mu=.5, mur=1., th_deg=20, v0=0.0, frames=240):
  r=.025; m=1600/.6*4/3*math.pi*r**3; k=20000.
  im=1/m; ii=2.5*im/(r*r); th=math.radians(th_deg)
  n=np.array([-math.sin(th),math.cos(th),0.]); g=np.array([0,-9.81,0.])
  tang=np.array([math.cos(th),math.sin(th),0.])
  h=(1/120)/S; alpha=1/(k*h*h)
  p=n*(r-5e-5); v=-tang*v0; w=np.zeros(3); along=[]
  for f in range(frames):
    for s in range(S):
      pred=p+h*(v+h*g); dx=np.zeros(3); dth=np.zeros(3); nsum=0
      pen=r-(p@n+(pred-p)@n); arm=-n*r
      if pen>0:
        lam=pen/(im+alpha); dx+=im*lam*n; nsum+=lam
        motion=(pred-p)+h*np.cross(w,arm); slip=motion-(motion@n)*n; sl=np.linalg.norm(slip)
        if sl>1e-12:
          tr=im; c=sl/tr; F=min(c,mu*lam)
          if F<=mur*lam:   # friction torque fits inside rolling resistance: translate
            dx-=im*slip*(F/sl)
          else:
            wt=3.5*im; lim=mu*lam
            corr=slip/wt if sl/wt<=lim else slip*(lim/sl)
            dx-=im*corr; dth-=ii*np.cross(arm,corr)
      xn=pred+dx; v=(xn-p)/h; w=w+dth/h
      sp=np.linalg.norm(w); cap=mur*nsum*r*ii/h
      if sp>0 and cap>0: w*=max(0,1-cap/sp)
      p=xn
    along.append(p@tang)
  return along, v, w
def main():
    for S in (20, 40, 80):
        a, _, _ = run(S)
        creep = abs(a[-1]-a[119])
        print('slope 20 deg mu_r=1 substeps', S, 'creep', creep)
        assert creep < 1e-9, 'held grain creeps'
    a, v, _ = run(20, v0=.5)
    assert abs(a[-1]-a[119]) < 1e-9, 'a grain landing on a holding slope must stop'
    # Rolling with resistance: a = g (sin - mu_r cos) / 1.4.
    a, _, _ = run(20, mur=.1)
    acc = 2*abs(a[-1]-a[0])/4
    expected = 9.81*(math.sin(math.radians(20))-.1*math.cos(math.radians(20)))/1.4
    print('rolling accel', acc, 'expected', expected)
    assert abs(acc-expected) < .05*expected
    # Sliding without rolling (mu_r > mu): a = g (sin - mu cos).
    a, _, _ = run(20, th_deg=30)
    acc = 2*abs(a[-1]-a[0])/4
    expected = 9.81*(math.sin(math.radians(30))-.5*math.cos(math.radians(30)))
    print('sliding accel', acc, 'expected', expected)
    assert abs(acc-expected) < .05*expected
    print('PASS xpbd grain friction reference')

if __name__ == '__main__':
    main()
