import copy


import numpy as np
from numpy.linalg import inv
import scipy as sp
import scipy.constants

# Internal modules
import utilities as Util
import kalmanfilter as KF
import datatypes


# ----------------------------------------------------------------------
class TrackFinder:
    def __init__(self, parameters=None, method="recursive", debug=False):
        self.method = method # {"recursive", "greedy"}
        self.debug = debug
        self.parameters={
            "cut_track_HitAddChi2": 30,          # Only used when method is "greedy"
            "cut_track_HitDropChi2": 10,         # Set to -1 to turn off
            "cut_track_HitProjectionSigma": 15,   # Number of sigmas
            "cut_track_TrackChi2Reduced": 7,
            "cut_track_TrackNHitsMin": 3,
            "fit_track_MultipleScattering": False
        }

    def run(self, hits):
        """
        Run all three rounds of kalman filter: find, drop, fit
        This function is a mixer of self.find() and self.filter_smooth()
        """
        self.seeds = self.seeding(hits)
        self.hits = copy.copy(hits)
        self.hits_grouped = Util.track.group_hits_by_layer(self.hits)
        self.tracks = []
        while len(self.seeds)>0:
            # ------------------------------------
            # Round 1: Find hits that belongs to one track
            seed = self.seeds[-1]; 
            if self.debug: print(f"--- New seed --- \n  seed: {seed}")
            hits_found, track_chi2 = self.find_once(self.hits, self.hits_grouped, seed)
            # Remove the current seed no matter the track is good or not:
            self.seeds.pop(-1)                
            # Apply cuts
            # If not enough hits, drop this track
            if len(hits_found)<self.parameters["cut_track_TrackNHitsMin"] or len(hits_found)==2:
                if self.debug: print(f"   finding failed, not enough hits. Hits found: {len(hits_found)}")
                continue            
            ndof = 3*len(hits_found) - 6
            track_chi2_reduced = track_chi2/ndof                   
            # If chi2 is too large, drop this track
            if track_chi2_reduced>self.parameters["cut_track_TrackChi2Reduced"]:
                if self.debug: print(f"   finding failed, chi2 too large. Chi2/nodf: {track_chi2}/{ndof}")
                continue


            # Sort the hits by time before running the filter
            hits_found.sort(key=lambda hit: hit.t)

            # ------------------------------------
            # Round 2: Run filter and smooth with the option to drop outlier during smoothing
            kalman_result, inds_dropped = self.filter_smooth(hits_found, drop_chi2=self.parameters["cut_track_HitDropChi2"])
            inds_dropped.sort(reverse=True)
            for ind in inds_dropped:
                hits_found.pop(ind)
            # If not enough hits, drop this track
            if len(hits_found)<self.parameters["cut_track_TrackNHitsMin"]:
                if self.debug: print(f"  fitting failed, not enough hits. Hits found: {len(hits_found)}")
                continue                

            # ------------------------------------
            # Round 3: Run filter and smooth again without dropping
            kalman_result, inds_dropped = self.filter_smooth(hits_found, drop_chi2=-1)  
            if self.debug: 
                print(f" Track found. Added hits:") 
                for t in hits_found:
                    print(t)


            # Remove other seeds that shares hits of the found track
            self.remove_related_hits_seeds(hits_found)

            # ------------------------------------
            # Finally, prepare the output
            track_output = self.prepare_output(kalman_result, hits_found, track_ind = len(self.tracks))  
            self.tracks.append(track_output)            



        return self.tracks



    def find(self, hits):
        self.seeds = self.seeding(hits)
        self.hits = copy.copy(hits)
        self.hits_grouped = Util.track.group_hits_by_layer(self.hits)
        self.tracks_found = []
        while len(self.seeds)>0:
            seed = self.seeds[-1]
            hits_found, track_chi2 = self.find_once(self.hits, self.hits_grouped, seed)


            # Remove the current seed no matter the track is good or not:
            self.seeds.pop(-1)         
            # Apply cuts
            # If not enough hits, drop this track
            if len(hits_found)<self.parameters["cut_track_TrackNHitsMin"]:
                continue            
            ndof = 3*len(hits_found) - 6
            if ndof<=0:
                continue            
            track_chi2_reduced = track_chi2/ndof                   
            # If chi2 is too large, drop this track
            if track_chi2_reduced>self.parameters["cut_track_TrackChi2Reduced"]:
                continue


            # Attach the track if it pass the cuts
            self.tracks_found.append(hits_found)

            # Remove other seeds that are in the track
            self.remove_related_hits_seeds(hits_found)
            

        return self.tracks_found


    def seeding(self, hits):
        """
        Find seed for tracks
        Returns a pair of index of the hit (not the hit itself!) and a score
        """
        c=sp.constants.c/1e7 # [cm/ns]
        seeds=[]
        for i in range(len(hits)):
            for j in range(i+1, len(hits)):
                if hits[i].y == hits[j].y:
                    continue
                dx = hits[i].x- hits[j].x
                dy = hits[i].y- hits[j].y
                dz = hits[i].z- hits[j].z
                dt = hits[i].t- hits[j].t
                ds = np.abs((dx**2+dy**2-dz**2)/c**2-dt**2)
                ds = ds/dt**2
                if ds>50:
                    continue
                seeds.append([i,j,ds])

        # Sort seeds by score
        # Reversed: place the best one at the end
        seeds.sort(key=lambda s: s[-1], reverse=True)
        return seeds

    def find_once(self, hits, hits_layer_grouped, seed):  
        #### General info ####
        LAYERS = np.sort(list(hits_layer_grouped.keys()))

        ##### Seed ####
        # Check the direction of seed by comparing the time of two hits
        seed_hits = [hits[seed[0]], hits[seed[1]]]
        # Alwayse have the first hit to be first in time
        if (seed_hits[0].t > seed_hits[1].t):
            seed_hits = seed_hits[::-1]
        seed_start_layer = seed_hits[0].layer
        seed_stop_layer  = seed_hits[1].layer        
        # Check if needed to find backward or forward
        FIND_FORWARD = True
        FIND_BACKWARD= True
        if (seed_hits[0].y > seed_hits[1].y):
            TRACK_DIRECTION = 0 # Downward track
            # FIND_FORWARD = seed_start_layer>=LAYERS[0]
            # FIND_FORWARD_LAYERS = LAYERS[:np.argmax(LAYERS>seed_stop_layer)-1][::-1]
            # FIND_BACKWARD = seed_stop_layer<LAYERS[-1]
            FIND_BACKWARD_LAYERS = LAYERS[np.argmax(LAYERS>seed_stop_layer):]     
        else:
            TRACK_DIRECTION = 1 # Upward track
            # FIND_FORWARD = seed_stop_layer<LAYERS[-1]
            # FIND_FORWARD_LAYERS  = LAYERS[np.argmax(LAYERS>seed_stop_layer):]
            # FIND_BACKWARD = seed_start_layer>=LAYERS[0]
            FIND_BACKWARD_LAYERS = LAYERS[:np.argmax(LAYERS>seed_stop_layer)-1][::-1]

        # print(FIND_FORWARD, FIND_FORWARD_LAYERS, FIND_BACKWARD, FIND_BACKWARD_LAYERS)
        


        ##### Find ####
        hits_found = [seed_hits[1]]
        chi2_found = 0
        if FIND_BACKWARD:
            step_pre = seed_hits[1].y # Keep track of the y of the previous step

            kf_find = KF.KalmanFilterFind()
            kf_find.init_filter(*Util.track.init_state(seed_hits))
            if self.method=="recursive":
                hits_found_backward, chi2 = self.find_in_layers_recursive(hits, hits_layer_grouped, FIND_BACKWARD_LAYERS, kf_find, step_pre)
            else:
                hits_found_backward, chi2 = self.find_in_layers_greedy(hits, hits_layer_grouped, FIND_BACKWARD_LAYERS, kf_find, step_pre, cut=False)
            # Order of found hits also needs to be reversed for backward finding
            hits_found_backward = hits_found_backward[::-1] 
            hits_found_backward.extend(hits_found)   
            hits_found = hits_found_backward           
        if FIND_FORWARD:
            if len(hits_found)<2:
                return [], []
            # Rest the seed to be the first two hits
            seed_hits = hits_found[:2]            
            step_pre = seed_hits[1].y # Keep track of the y of the previous step
            if (seed_hits[0].y > seed_hits[1].y):
                FIND_FORWARD_LAYERS = LAYERS[:np.argmax(LAYERS>seed_hits[1].layer)-1][::-1]
            else:
                FIND_FORWARD_LAYERS  = LAYERS[np.argmax(LAYERS>seed_hits[1].layer):]            

            kf_find = KF.KalmanFilterFind()
            kf_find.init_filter(*Util.track.init_state(seed_hits)) # Set initial state using two hits specified by the seed
            if self.method=="recursive":
                hits_found_forward,chi2 = self.find_in_layers_recursive(hits, hits_layer_grouped, FIND_FORWARD_LAYERS, kf_find, step_pre)
            else:
                hits_found_forward,chi2 = self.find_in_layers_greedy(hits, hits_layer_grouped, FIND_FORWARD_LAYERS, kf_find, step_pre)
            chi2_found+=chi2
            hits_found = hits_found[:2] + hits_found_forward
   

        return hits_found,chi2_found

    def find_in_layers_greedy(self, hits, hits_layer_grouped, layers_to_scan, kf_find, step_pre, cut=True):
        """
        Find the hit that has minimum chi2 in each layer
        """
        hits_found = []
        for layer in layers_to_scan:
            hits_thislayer = hits_layer_grouped[layer]
            if len(hits_thislayer)==0:
                continue

            # Get one hit
            hit = hits_thislayer[0]

            # Get the prediction matrix 
            step_this = hits_thislayer[0].y 
            dy = step_this - step_pre # Step size

            Ax, Az, At = kf_find.Xf[3:]
            velocity = [Ax/At, 1/At, Az/At]     if self.parameters["fit_track_MultipleScattering"] else None       # Velocity is needed for multiple scattering            
            _, Vi, Hi, Fi, Qi = Util.track.add_measurement(hits_thislayer[0], dy, velocity=velocity) # Calculate matrices. Only need to do once for all this in the same layer
            kf_find.update_matrix(Vi, Hi, Fi, Qi) # pass matrices to KF

            # Use the Predicted location to limit the search range
            Xp = kf_find.Xp_i
            Xp_unc = np.sqrt(np.diag(kf_find.Rp_i))
            # Function to test if new measurement is within N_sigma times the uncertainty ellipsoid
            N_sigma = self.parameters["cut_track_HitProjectionSigma"]
            # Use the total uncertainty of the prediction plus the measurement
            unc_total = [np.linalg.norm([hit.x_err,Xp_unc[0]]), np.linalg.norm([hit.z_err,Xp_unc[1]]), np.linalg.norm([hit.t_err,Xp_unc[2]])]
            test_measurement_incompatible = lambda x,z,t: abs(x-Xp[0])>unc_total[0]*N_sigma or \
                                                        abs(z-Xp[1])>unc_total[1]*N_sigma or \
                                                        abs(t-Xp[2])>unc_total[2]*N_sigma or \
                                                        ((x-Xp[0])/unc_total[0])**2 + ((z-Xp[1])/unc_total[1])**2 + ((t-Xp[2])/unc_total[2])**2 > N_sigma**2            

            # Calculate chi2 for all hits in the next layer
            # chi2_predict = [kf_find.forward_predict_chi2(np.array([mi.x, mi.z, mi.t])) for mi in hits_thislayer]
            chi2_predict=[]
            chi2_predict_inds =[]
            for imeasurement, m in enumerate(hits_thislayer):
                if test_measurement_incompatible(m.x, m.z, m.t):
                    continue
                else:
                # if 1:
                    chi2 = kf_find.forward_predict_chi2(np.array([m.x, m.z, m.t]))
                    chi2_predict.append(chi2)
                    chi2_predict_inds.append(imeasurement)
            if len(chi2_predict)==0:
                return [],0


            # Find the hit with minimum chi2
            chi2_min_idx = np.argmin(chi2_predict)
            if chi2_predict[chi2_min_idx]<self.parameters["cut_track_HitAddChi2"] or cut==False:
                # Save the hit either if the chi2 is lower than the threshold, or the cut is disabled
                hits_found.append(hits_thislayer[chi2_predict_inds[chi2_min_idx]])
                # Update the step and the Kalman filter
                step_pre = step_this
                mi = hits_found[-1]
                kf_find.forward_filter(np.array([mi.x, mi.z, mi.t]))
                if self.debug: print("  Hit found:", mi, "; chi2", chi2_predict[chi2_min_idx])
            else:
                if self.debug: print(f"  No hits added from layer {layer}. Chi2 of hits {chi2_predict}. Hits", np.array(hits_thislayer)[chi2_predict_inds])


        return hits_found, kf_find.chift_total


    def find_in_layers_recursive(self, hits, hits_layer_grouped, layers_to_scan, kf_find, step_pre):
        """
        Find the hits that has minimum chi2 in total
        """
        self.found_hit_groups = []
        self.found_chi2_groups = []

        # Run the recrusive finding
        current_layer_ind = -1
        found_hits_inds = []
        found_chi2s = []
        self._find_in_layers_recursive(hits_layer_grouped, layers_to_scan, kf_find, step_pre, current_layer_ind, found_hits_inds, found_chi2s)

        if len(self.found_hit_groups)==0:
            return [], 0

        # Find the group with minimum chi2 per hit
        n_hits = [max(len(i),1) for i in self.found_hit_groups] # limit to be at least 1 to not mess up the divide in the following line
        chi2_reduced = np.sum(self.found_chi2_groups, axis=1)/n_hits
        ind_minchi2 = np.argmin(chi2_reduced)
        hits_found = [hits[i] for i in self.found_hit_groups[ind_minchi2] ]
        return hits_found, chi2_reduced[ind_minchi2]
        
    def _find_in_layers_recursive(self, hits_layer_grouped, layers_to_scan, kf_find, step_pre, current_layer_ind, found_hits_inds, found_chi2s):
        current_layer_ind+=1 

        if current_layer_ind>=len(layers_to_scan):
            self.found_hit_groups.append(found_hits_inds)
            self.found_chi2_groups.append(found_chi2s)
            return

        layer = layers_to_scan[current_layer_ind]
        hits_thislayer = hits_layer_grouped[layer]

        # Get the prediction matrix 
        step_this = hits_thislayer[0].y 
        dy = step_this - step_pre # Step size
        step_pre = step_this
        Ax, Az, At = kf_find.Xf[3:]
        velocity = [Ax/At, 1/At, Az/At]     if self.parameters["fit_track_MultipleScattering"] else None   # Velocity is needed for multiple scattering        
        _, Vi, Hi, Fi, Qi = Util.track.add_measurement(hits_thislayer[0], dy, velocity=velocity) # Calculate matrices. Only need to do once for all this in the same layer
        kf_find.update_matrix(Vi, Hi, Fi, Qi) # pass matrices to KF

        # Predicted location 
        Xp = kf_find.Xp_i
        Xp_unc = np.sqrt(np.diag(kf_find.Rp_i))
        # Function to test if new measurement is within N_sigma times the uncertainty ellipsoid
        N_sigma = self.parameters["cut_track_HitProjectionSigma"]
        # Use the total uncertainty of the prediction plus the measurement
        hit = hits_thislayer[0]
        unc_total = [np.linalg.norm([hit.x_err,Xp_unc[0]]), np.linalg.norm([hit.z_err,Xp_unc[1]]), np.linalg.norm([hit.t_err,Xp_unc[2]])]
        test_measurement_incompatible = lambda x,z,t: abs(x-Xp[0])>unc_total[0]*N_sigma or \
                                                    abs(z-Xp[1])>unc_total[1]*N_sigma or \
                                                    abs(t-Xp[2])>unc_total[2]*N_sigma or \
                                                    ((x-Xp[0])/unc_total[0])**2 + ((z-Xp[1])/unc_total[1])**2 + ((t-Xp[2])/unc_total[2])**2 > N_sigma**2    

        # calculate chi2 for all hits in the next layer
        for imeasurement, m in enumerate(hits_thislayer):
            # Limit our search to the hits close to prediction:
            if test_measurement_incompatible(m.x, m.z, m.t):
                continue
            # print(m.x, m.z, m.t)
            # print(Xp[:3])
            # print(test_measurement_compatible(m.x, m.z, m.t))

            # Make copys for hits and chi2s for each recursion
            found_hits_inds_i = copy.deepcopy(found_hits_inds)
            found_chi2s_i = copy.deepcopy(found_chi2s)            
            kf_find_i = copy.deepcopy(kf_find)
            # Run Kalman filter
            chi2 = kf_find_i.forward_filter(np.array([m.x, m.z, m.t]))
            found_hits_inds_i.append(m.ind)
            found_chi2s_i.append(chi2)
            self._find_in_layers_recursive(hits_layer_grouped, layers_to_scan, kf_find_i, step_pre, current_layer_ind, found_hits_inds_i, found_chi2s_i)
                
        return

    def remove_related_hits_seeds(self, hits_found):
        hits_found_inds = [hit.ind for hit in hits_found]
        hits_found_inds.sort(reverse=True)
        # Remove seeds 
        # Need to do backwards to not change the index
        for i in reversed(range(len(self.seeds))):
            seed = self.seeds[i]
            if (seed[0] in hits_found_inds) or (seed[1] in hits_found_inds):
                self.seeds.pop(i)

        # Redo the grouping
        for layer in list(self.hits_grouped.keys()):
            hits = self.hits_grouped[layer]
            for ihit in reversed(range(len(hits))):
                if hits[ihit].ind in hits_found_inds:
                    hits.pop(ihit)

            if len(hits)==0:
                self.hits_grouped.pop(layer)



    def filter_smooth(self, hits, drop_chi2=-1):
        """
        Run the forward filter and backward smooth at once
        
        INPUT
        ---
        hits: list
            A list of all hits in a track
        drop_chi2: float
            for values less than zero, disable dropping
            for values zero, the steps with chi2 larger than this number will be dropped
        """
        kf = KF.KalmanFilter()

        # Set initial state using first two hits
        m0, V0, H0, Xf0, Cf0, Rf0 = Util.track.init_state(hits) # Use the first two hits to initiate
        kf.init_filter( m0, V0, H0, Xf0, Cf0, Rf0)
        

        # Feed all measurements to KF
        for i in range(2,len(hits)):   
            # get updated matrix
            hit = hits[i]
            dy  = hits[i].y-hits[i-1].y

            Ax, Az, At = kf.Xf[-1][3:]
            velocity = [Ax/At, 1/At, Az/At]     if self.parameters["fit_track_MultipleScattering"] else None        # Velocity is needed for multiple scattering
            mi, Vi, Hi, Fi, Qi = Util.track.add_measurement(hit, dy, velocity=velocity)
            
            # pass to KF
            kf.forward_predict(mi, Vi, Hi, Fi, Qi)
            kf.forward_filter()

        # Filter backward
        dropped_inds = []
        if drop_chi2<0:
            kf.backward_smooth()
        else:
            # Manually go through all steps to check if it exceed drop_chi2
            kf.init_smooth()
            while kf.CURRENT_STEP>=0:
                chi2_temp = kf.smooth_step_try()

                dropped =  chi2_temp>drop_chi2
                if dropped:
                    dropped_inds.append(kf.CURRENT_STEP)
                    if self.debug: print(f"   hit dropped with chi2 {chi2_temp}. Hit {hits[kf.CURRENT_STEP][:6]}")
                # Finishing the current step
                kf.smooth_step(drop = dropped)

        return kf, dropped_inds 

    

    def prepare_output(self, kalman_result, hits_found, track_ind=0):
        """ 
        Turn the Kalman filter result into a Track object

        """
        # propagate the KF result from the second hit to the first hit
        Ax, Az, At = kalman_result.Xsm[0][3:]
        velocity = [Ax/At, 1/At, Az/At]     if self.parameters["fit_track_MultipleScattering"] else None       # Velocity is needed for multiple scattering      

        mi, Vi, Hi, Fi, Qi = Util.track.add_measurement(hits_found[0], hits_found[0].y - hits_found[1].y, velocity=velocity)
        state_predicted_step_0 = Fi@kalman_result.Xsm[0]
        statecov_predicted_step_0 = Fi@kalman_result.Csm[0]@Fi.T + Qi 
        # Add the covariance of one additional layer:
        cov = statecov_predicted_step_0 + Qi
        chi2 = kalman_result.chift_total
        ind = track_ind
        hits = [hit.ind for hit in hits_found]


        
        x0 = state_predicted_step_0[0]
        z0 = state_predicted_step_0[1]
        t0 = state_predicted_step_0[2]
        Ax = state_predicted_step_0[3]
        Az = state_predicted_step_0[4]
        At = state_predicted_step_0[5]

        y0 = hits_found[0].y
        Ay = 1 # Slope of Y vs Y, which is always 1

        hits_filtered = [[xsm[0], hit.y, xsm[1], xsm[2]] for hit,xsm in zip(hits_found[1:], kalman_result.Xsm)]
        hits_filtered.insert(0, [x0,y0,z0,t0])

        # Track is a namedtuple("Track", ["x0", "y0", "z0", "t", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
        track = datatypes.Track(x0, y0, z0, t0, Ax, Ay, Az, At, cov, chi2, ind, hits, hits_filtered)
        return track