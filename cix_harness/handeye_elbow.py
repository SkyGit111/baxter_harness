"""Continue along Baxter's one-dimensional IK solution family at a fixed tool pose."""
import numpy as np


def redundant_branches(ik, goal, start, seeds, margin):
    if not seeds:
        return []
    k = ik.kdl
    jac_solver = k.ChainJntToJacSolver(ik.chain)
    jac = k.Jacobian(7)
    names = ik.names
    low = np.array([ik.lower[i]+margin for i in range(7)])
    high = np.array([ik.upper[i]-margin for i in range(7)])
    array = lambda q: np.array([q[n] for n in names])
    def joint(values):
        ans=k.JntArray(7)
        for i,v in enumerate(values):
            ans[i]=float(v)
        return ans
    # Near a single root, ordinary seed perturbations repeatedly converge to
    # that same elbow posture. March tangentially along the Jacobian null space
    # and correct the residual against exactly the same Cartesian goal.
    roots=[]
    for seed in seeds:
        values=array(seed)
        if all(np.max(np.abs(values-r))>.60 for r in roots):
            roots.append(values)
    candidates=[]
    for root in roots[:4]:
        for direction in (-1.,1.):
            current=root.copy()
            tangent=None
            for _ in range(100):
                if jac_solver.JntToJac(joint(current),jac)<0:
                    raise RuntimeError('Unable to compute the Baxter arm Jacobian')
                J=np.array([[jac[i,j] for j in range(7)] for i in range(6)])
                _,singular,vh=np.linalg.svd(J,full_matrices=True)
                if not np.isfinite(singular).all() or singular[-1]<1e-6:
                    break
                vec=vh[-1]
                if tangent is None:
                    vec=vec*direction*np.sign(vec[np.argmax(abs(vec))])
                elif vec@tangent<0:
                    vec=-vec
                probe=current+.045*vec
                if np.any(probe<low) or np.any(probe>high):
                    break
                out=k.JntArray(7)
                if ik.solver.CartToJnt(joint(probe),goal,out)<0:
                    break
                solution=np.array([out[i] for i in range(7)])
                delta=solution-current
                if (not np.isfinite(solution).all() or np.any(solution<low) or np.any(solution>high)
                        or delta@vec<.010 or np.linalg.norm(delta)>.12):
                    break
                if not any(np.max(np.abs(solution-q))<.020 for q in candidates):
                    candidates.append(solution)
                current=solution;tangent=vec
    ordered=sorted(candidates,key=lambda q:float(np.linalg.norm(q-array(start))))
    return [{n:float(v) for n,v in zip(names,q)} for q in ordered]
