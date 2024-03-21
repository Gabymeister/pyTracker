import numpy as np
from numpy.linalg import inv
import KalmanFilter as KF

def init_state(hits):
    """m0, V0, H0, Xf0, Cf0, Rf0"""
    dt=hits[1].t-hits[0].t
    dx=hits[1].x-hits[0].x
    dy=hits[1].y-hits[0].y
    dz=hits[1].z-hits[0].z
    
    # Initial State Vector X0
    Xf0 = np.array([hits[1].x, hits[1].z, hits[1].t, dx/dy, dz/dy, dt/dy])
    
    # Initial Covariance P0
    J =np.array([[ 0       , 0           , 0       , 0       , 1       , 0             , 0     , 0     ],
                 [ 0       , 0           , 0       , 0       , 0       , 0             , 1     , 0     ],
                 [ 0       , 0           , 0       , 0       , 0       , 0             , 0     , 1     ],
                 [- 1 / dy, dx / (dy*dy) , 0       , 0       , 1 / dy  , - dx / (dy*dy), 0     , 0     ],
                 [0       , dz / (dy*dy) , - 1 / dy, 0       , 0       , - dz / (dy*dy), 1 / dy, 0     ],
                 [0       , dt / (dy*dy) , 0       , - 1 / dy, 0       , - dt / (dy*dy), 0     , 1 / dy]])

    err0=np.diag([hits[0].x_err, hits[0].y_err, hits[0].z_err, hits[0].t_err,
                  hits[1].x_err, hits[1].y_err, hits[1].z_err, hits[1].t_err])**2
    Cf0=J.dot(err0).dot(J.T)
    
    # the rest
    m0 = np.array([hits[1].x, hits[1].z, hits[1].t])
    V0  = np.diag([hits[1].x_err, hits[1].z_err, hits[1].t_err])**2
    Rf0 = np.diag([hits[1].x_err, hits[1].z_err, hits[1].t_err])**2
    H0 = np.array([[1,0,0,0,0,0],
                   [0,1,0,0,0,0],
                   [0,0,1,0,0,0]])
    
    return m0, V0, H0, Xf0, Cf0, Rf0



def add_measurement(hit, dy):
    """
    mi, Vi, Hi, Fi, Qi
    """
    mi = np.array([hit.x, hit.z, hit.t])
    Vi = np.diag([hit.x_err, hit.z_err, hit.t_err])**2
    Hi = np.array([[1,0,0,0,0,0],
                   [0,1,0,0,0,0],
                   [0,0,1,0,0,0]])
    Fi = np.array([[1, 0, 0,dy, 0, 0],
                   [0, 1, 0, 0,dy, 0],
                   [0, 0, 1, 0, 0,dy],
                   [0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1]])
    Qi = update_Q()
    return mi, Vi, Hi, Fi, Qi

def update_Q():

    
    return 0


def run_kf(hits):
    kf = KF.KalmanFilter()

    # Set initial state using first two hits
    m0, V0, H0, Xf0, Cf0, Rf0 = init_state(hits) # Use the first two hits to initiate
    kf.init_filter( m0, V0, H0, Xf0, Cf0, Rf0)
    

    # Feed all measurements to KF
    for i in range(2,len(hits)):   
        # get updated matrix
        hit = hits[i]
        dy  = hits[i].y-hits[i-1].y
        mi, Vi, Hi, Fi, Qi = add_measurement(hit, dy)
        
        # pass to KF
        kf.forward_predict(mi, Vi, Hi, Fi, Qi)
        kf.forward_filter()

    # Filter backward
    kf.backward_smooth()
    return kf    


def group_hits_by_layer(hits):
    # Assign a unique index to hits
    for ihit in range(len(hits)):
        hits[ihit] = hits[ihit]._replace(ind = ihit)

    # Layers
    layers = np.unique([hit.layer for hit in hits])
    HitsLayerGrouped={layer:[] for layer in layers}
    for ihit in range(len(hits)):
        hit = hits[ihit]
        HitsLayerGrouped[hit.layer].append(hit)
    return HitsLayerGrouped    