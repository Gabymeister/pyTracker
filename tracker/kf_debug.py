import numpy as np
import scipy as sp
from scipy import constants
from collections import namedtuple

from iminuit import Minuit
import iminuit

from DataTypes import *

def make_hits(x, y, z, t):
    Y_LAYERS = y
    
    det_width  = 4.5 # 4.5cm per bar
    det_height = 1 #[cm]
    time_resolution = 1 #[ns], single channel
    refraction_index = 1.5
    
    unc_trans = det_width/np.sqrt(12)                  
    unc_long = time_resolution*constants.c/1e7/np.sqrt(2)/refraction_index
    UNC_T = time_resolution # ns
    UNC_Y = det_height/np.sqrt(12) # uncertainty in thickness, cm
    

    hits=[]
    for i in Y_LAYERS:
        if i%2==1:
            hits.append(Hit(x[i], y[i], z[i], t[i], unc_trans, 0, unc_long, UNC_T, i, i))
        else:
            hits.append(Hit(x[i], y[i], z[i], t[i], unc_long, 0, unc_trans, UNC_T, i, i))         
            
    return hits

def gen_hits(x0=0,y0=0,z0=0, t0=0, Ax=0.3,Az=0.2,At=1/28, N_LAYERS = 8):
    Y_LAYERS = 12_00 + np.arange(N_LAYERS)*80
    
    det_width  = 4.5 # 4.5cm per bar
    det_height = 1 #[cm]
    time_resolution = 1 #[ns], single channel
    refraction_index = 1.5
    
    unc_trans = det_width/np.sqrt(12)                  
    unc_long = time_resolution*constants.c/1e7/np.sqrt(2)/refraction_index
    UNC_T = time_resolution # ns
    UNC_Y = det_height/np.sqrt(12) # uncertainty in thickness, cm
    

    hits=[]
    hits_truth=[]
    for i in range(N_LAYERS):
        dy = Y_LAYERS[i]-Y_LAYERS[0]
        hits_truth.append(Hit(x0 + Ax*dy, Y_LAYERS[i], z0 + Az*dy, t0 + At*dy , 0, 0, 0, 0, i, i))
        

        if i%2==1:
            hits.append(Hit(hits_truth[-1].x//det_width*det_width,
                            hits_truth[-1].y,
                            hits_truth[-1].z+np.random.normal(0,unc_long),
                            hits_truth[-1].t+np.random.normal(0,UNC_T),
                           unc_trans, 0, unc_long, UNC_T, i, i))
        else:
            hits.append(Hit(hits_truth[-1].x+np.random.normal(0,unc_long),
                            hits_truth[-1].y,
                            hits_truth[-1].z//det_width*det_width,
                            hits_truth[-1].t+np.random.normal(0,UNC_T),
                           unc_long, 0, unc_trans, UNC_T, i, i))         
            
        
    par_truth = [x0,z0,t0,Ax,Az,At]
        
    return hits, hits_truth, np.array(par_truth)



# -------------------------------------
# LS fit
# ------------------------------------
class chi2_track:
    def __init__(self, hits):
        self.hits=hits
        self.func_code = iminuit.util.make_func_code(['x0', 'z0', 't0', 'Ax', 'Az', 'At'])
    def __call__(self, x0, z0, t0, Ax, Az, At):
        error=0
        y0 = 10
        for hit in self.hits:
            dy = (hit.y - self.hits[1].y)
            model_x = x0 + Ax*dy
            model_z = z0 + Az*dy
            model_t = t0 + At*dy
            error+= np.sum(np.power([(model_t-hit.t)/hit.t_err, 
                                     (model_x-hit.x)/hit.x_err, 
                                     (model_z-hit.z)/hit.z_err],2))
        return error        

def guess_track(hits):
    # Guess initial value
    x0_init = hits[0].x
    z0_init = hits[0].z
    t0_init = hits[0].t
    
    dy=hits[-1].t-hits[0].t
    Ax_init = (hits[-1].x-hits[0].x)/dy
    Az_init = (hits[-1].z-hits[0].z)/dy
    At_init = (hits[-1].t-hits[0].t)/dy
    return  (x0_init, z0_init,t0_init,Ax_init,Az_init,At_init)
    
def fit_track(hits, guess):
    x0_init, z0_init,t0_init,Ax_init,Az_init,At_init = guess

    m = Minuit(chi2_track(hits),x0=x0_init, z0=z0_init, t0=t0_init, Ax=Ax_init,Az=Az_init, At=At_init)
    # m.fixed["y0"]=True
    m.limits["x0"]=(-100000,100000)
    m.limits["z0"]=(-100000,100000)
    m.limits["t0"]=(-100,1e5)
    m.limits["Ax"]=(-10,10) # Other
    m.limits["Az"]=(-10,10)
    m.limits["At"]=(0.001,0.2) # Beam direction; From MKS unit to cm/ns = 1e2/1e9=1e-7
    m.errors["x0"]=0.1
    m.errors["z0"]=0.1
    m.errors["t0"]=0.1
    m.errors["Ax"] = 0.001
    m.errors["At"] = 0.0001
    m.errors["Az"] = 0.001

    m.migrad()  # run optimiser
    m.hesse()   # run covariance estimator
    
    return m