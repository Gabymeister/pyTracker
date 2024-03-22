from collections import namedtuple

import numpy as np
from numpy.linalg import inv
import scipy as sp
import iminuit


import kalmanfilter as KF
import datatypes

class hit:
    @staticmethod
    def make_hits(x, y, z, t, ylayers):
        Y_LAYERS = ylayers
        det_width  = 4.5 # 4.5cm per bar
        det_height = 1 #[cm]
        time_resolution = 1 #[ns], single channel
        refraction_index = 1.5
        
        unc_trans = det_width/np.sqrt(12)                  
        unc_long = time_resolution*sp.constants.c/1e7/np.sqrt(2)/refraction_index
        UNC_T = time_resolution # ns
        UNC_Y = det_height/np.sqrt(12) # uncertainty in thickness, cm
        

        hits=[]
        for i, layer in enumerate(Y_LAYERS):
            if i%2==1:
                hits.append(datatypes.Hit(x[i], y[i], z[i], t[i], unc_trans, 0, unc_long, UNC_T, layer, i))
            else:
                hits.append(datatypes.Hit(x[i], y[i], z[i], t[i], unc_long, 0, unc_trans, UNC_T, layer, i))         
                
        return hits

    @staticmethod
    def gen_hits(x0=0,y0=0,z0=0, t0=0, Ax=0.3,Az=0.2,At=1/28, N_LAYERS = 8):
        Y_LAYERS = 12_00 + np.arange(N_LAYERS)*80
        
        det_width  = 4.5 # 4.5cm per bar
        det_height = 1 #[cm]
        time_resolution = 1 #[ns], single channel
        refraction_index = 1.5
        
        unc_trans = det_width/np.sqrt(12)                  
        unc_long = time_resolution*sp.constants.c/1e7/np.sqrt(2)/refraction_index
        UNC_T = time_resolution # ns
        UNC_Y = det_height/np.sqrt(12) # uncertainty in thickness, cm
        

        hits=[]
        hits_truth=[]
        for i in range(N_LAYERS):
            dy = Y_LAYERS[i]-Y_LAYERS[0]
            hits_truth.append(datatypes.Hit(x0 + Ax*dy, Y_LAYERS[i], z0 + Az*dy, t0 + At*dy , 0, 0, 0, 0, i, i))
            

            if i%2==1:
                hits.append(datatypes.Hit(hits_truth[-1].x//det_width*det_width,
                                hits_truth[-1].y,
                                hits_truth[-1].z+np.random.normal(0,unc_long),
                                hits_truth[-1].t+np.random.normal(0,UNC_T),
                            unc_trans, 0, unc_long, UNC_T, i, i))
            else:
                hits.append(datatypes.Hit(hits_truth[-1].x+np.random.normal(0,unc_long),
                                hits_truth[-1].y,
                                hits_truth[-1].z//det_width*det_width,
                                hits_truth[-1].t+np.random.normal(0,UNC_T),
                            unc_long, 0, unc_trans, UNC_T, i, i))         
                
            
        par_truth = [x0,z0,t0,Ax,Az,At]
            
        return hits, hits_truth, np.array(par_truth)



class track:
    @staticmethod
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


    @staticmethod
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
        Qi = track.update_Q()
        return mi, Vi, Hi, Fi, Qi

    @staticmethod
    def update_Q():
        # dy=100
        # a=0.1
        # c=0.1
        # b=np.sqrt(1-a**2-c**2)
        # mag=30

        # b2 = b**2
        # a2 = a**2
        # dy2 = dy*dy
        # b4 = np.power(b, 4)
        # mag2 = np.power(mag , 2)

        # Q=np.array([[dy2 * (b2 +a2) / b4,
        #     dy2 * a / (mag * b4),
        #     dy2 * a * c / b4,
        #     mag  * dy / b,
        #     -mag  * dy * a / (b2),
        #     0,],
        #     [dy2 * a / (mag  * b4),
        #     dy2 * (1 - b2) / (mag2 * b4),
        #     dy2 * c / (mag  * b4),
        #     dy * a / b,
        #     -dy * (1 - b2) / (b2),
        #     dy * c / b,],
        #     [dy2 * a * c / b4,
        #     dy2 * c / (mag  * b4),
        #     dy2 * (c * c + b2) / b4,
        #     0,
        #     -mag  * dy * c / (b2),
        #     mag  * dy / b,],
        #     [mag  * dy / b,
        #     dy * a / b,
        #     0,
        #     mag2 * (1 -a2),
        #     -mag2 * (a * b),
        #     -mag2 * (a * c),],
        #     [-mag  * dy * a / (b2),
        #     -dy * (1 - b2) / (b2),
        #     -mag  * dy * c / (b2),
        #     -mag2 * (a * b),
        #     mag2 * (1 - b2),
        #     -mag2 * (b * c),],
        #     [0,
        #     dy * c / b,
        #     mag  * dy / b,
        #     -mag2 * (a * c),
        #     -mag2 * (b * c),
        #     mag2 * (1 - c * c)]])

        Q=0
        return Q


    @staticmethod
    def run_kf(hits):
        kf = KF.KalmanFilter()

        # Set initial state using first two hits
        m0, V0, H0, Xf0, Cf0, Rf0 = track.init_state(hits) # Use the first two hits to initiate
        kf.init_filter( m0, V0, H0, Xf0, Cf0, Rf0)
        

        # Feed all measurements to KF
        for i in range(2,len(hits)):   
            # get updated matrix
            hit = hits[i]
            dy  = hits[i].y-hits[i-1].y
            mi, Vi, Hi, Fi, Qi = track.add_measurement(hit, dy)
            
            # pass to KF
            kf.forward_predict(mi, Vi, Hi, Fi, Qi)
            kf.forward_filter()

        # Filter backward
        kf.backward_smooth()
        return kf    


    @staticmethod
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

    @staticmethod
    def chi2_point_track(point, track, point_unc=None):
        """ 
        Calculate the chi-squre distance between 
        a 4-D point [x,y,z,t] and a track parameterized by [x0, y0, z0, t0, Ax, Az, At]
        where Ax = dx/dy, Az = dz/dy, At = dt/dy

        INPUT:
        ---
        point: list
            [x,y,z,t]
        track: namedtuple
            namedtuple("Track", ["x0", "y0", "z0", "t0", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
        point_unc: 1d-list or 2d-list or None
            [x_err,y_err,z_err,t_err]
        
        RETURN:
        ---
        chi2: float
            chi-square distance between the point and the track

        TEST:
        ```
        track1 = datatypes.Track(0,0,0, 0, 1,1,0,1, np.diag(np.ones(6)), 0,0,0,0)
        track2 = datatypes.Track(0,0,1, 0, -1,1,0,1, np.diag(np.ones(6)), 0,0,0,0)
        midpoint,dist = Util.track.closest_approach_midpoint_Track(track1, track2)
        chi2_point_track(midpoint, track2)

        0.25
        ```
        """
        x,y,z,t = point
        dy =  y - track.y0

        # Residual
        track_x = track.x0 + track.Ax*dy
        track_z = track.z0 + track.Az*dy
        track_t = track.t0 + track.At*dy  
        residual = np.array([track_x-x, track_z-z, track_t-t])

        # Covariance
        jac=np.array([[ 	1,  0,	0,  dy,   0,    0],
                        [ 	0,  1,  0,   0,  dy,    0],
                        [	0,  0, 	1,   0,   0,    dy]])
        covariance = jac @ track.cov @ jac.T

        # Add the uncertainty of the point
        if point_unc is not None:
            if np.array(point_unc)==1:
                x_err,y_err,z_err,t_err = point_unc
                covariance += np.diag(np.array([x_err, z_err, t_err])**2)
            elif np.array(point_unc)==2:
                covariance += point_unc

        chi2 = residual.T @ np.linalg.inv(covariance) @ residual

        return chi2    


    @staticmethod
    def closest_approach_midpoint(tr1, tr2):
        """
        INPUT:
        ---
        tr1,tr2: list
            ["x0", "y0", "z0", "vx", "vy", "vz", "t"])
            
        return:
        ---
        midpoint([x,y,z,t]), distance

        Test
        ```
        tr1 = np.array([0,0,0, 1, 0, 0, 0])
        tr2 = np.array([0,0,1, 0, 1, 0, 0])
        Util.track.closest_approach_midpoint(tr1,tr2)
        return: (array([0. , 0. , 0.5, 0. ]), 1.0)
        """
        

        rel_v = tr2[3:6] - tr1[3:6]
        rel_v2 = np.dot(rel_v, rel_v) 

        # Find the time at midpoint
        displacement = tr1[:3] - tr2[:3]; # position difference
        t_ca = (  np.dot(displacement, rel_v) + np.dot((tr2[3:6]*tr2[6] - tr1[3:6]*tr1[6]), rel_v)  )/rel_v2    

        pos1 = tr1[:3] + tr1[3:6]*(t_ca - tr1[6])
        pos2 = tr2[:3] + tr2[3:6]*(t_ca - tr2[6])
        midpoint = (pos1 + pos2)*(0.5)
        midpoint = np.append(midpoint, t_ca)
        
        distance = np.linalg.norm((pos1- pos2))
        return midpoint, distance

    @staticmethod
    def closest_approach_midpoint_Track(track1, track2):
        """
        INPUT:
        ---
        track1,track2: namedtuple
            namedtuple("Track", ["x0", "y0", "z0", "t0", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
            
        return:
        ---
        midpoint([x,y,z,t]), distance
        """
        tr1 = np.array([track1.x0, track1.y0, track1.z0, track1.Ax/track1.At, track1.Ay/track1.At, track1.Az/track1.At, track1.t0])
        tr2 = np.array([track2.x0, track2.y0, track2.z0, track2.Ax/track2.At, track2.Ay/track2.At, track2.Az/track2.At, track2.t0])

        return track.closest_approach_midpoint(tr1,tr2)


    @staticmethod
    def line_distance(tr1,tr2,time):
        """ 
        Calculate the distance of two lines at a certain time
        INPUT:
        ---
        tr1,tr2: list
            ["x0", "y0", "z0", "vx", "vy", "vz", "t0"])
        time: float
            the time to calculate distance
        """
        pos1 = tr1[:3] + tr1[3:6]*(t_ca - tr1[6])
        pos2 = tr2[:3] + tr2[3:6]*(t_ca - tr2[6])  
        displacement = pos1-pos2
        
        return np.dot(displacement,displacement) 

    @staticmethod
    def position(track, t=None, y=None, x=None, z=None) :
        if y is not None:
            dy = y-track.y0
            x = track.x0 + track.Ax*dy
            z = track.z0 + track.Az*dy
            t = track.t0 + track.At*dy 
        elif t is not None:
            dt = t-track.y0
            x = track.x0 + track.Ax/track.At*dy
            y = track.y0 + 1/track.At*dy             
            z = track.z0 + track.Az/track.At*dy
        return np.array([x,y,z,t])


    @staticmethod
    def distance_to_point(line, point):
        track_position = track.position(line, y=point[1])
        return np.linalg.norm((track_position-point)[:3])         


    @staticmethod
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
        
    @staticmethod
    def fit_track(hits, guess):
        x0_init, z0_init,t0_init,Ax_init,Az_init,At_init = guess

        m = iminuit.Minuit(chi2_track(hits),x0=x0_init, z0=z0_init, t0=t0_init, Ax=Ax_init,Az=Az_init, At=At_init)
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
       

class vertex:
    a=1
