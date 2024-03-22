
import copy
from collections import namedtuple

import numpy as np
from numpy.linalg import inv
import scipy as sp
import scipy.constants
import iminuit


# Internal modules
import utilities as Util
import kalmanfilter as KF
import datatypes



class chi2_vertex:
    def __init__(self, tracks):
        self.tracks=tracks
        self.func_code = iminuit.util.make_func_code(['x0', 'y0', 'z0', 't0'])
    def __call__(self, x0, y0, z0, t0):
        error=0
        point = [x0, y0, z0, t0]
        for track in self.tracks:
            error += Util.track.chi2_point_track(point, track)
        return error        

class VertexFitter:
    def __init__(self, parameters=None, method="ls", debug=False):
        self.debug = debug
        self.method = method # {"ls", "kalman"}
        self.trackinfo =  namedtuple("trackinfo",["track_ind", "track_chi2", "track_vertex_chi2", "track_vertex_dist"])
        self.parameters={
            "cut_vertex_SeedDist": 300,          # [cm]
            "cut_vertex_SeedChi2": 20,          
            "cut_vertex_TrackChi2Reduced": 30,        # NOT USED
            "cut_vertex_TrackAddDist": 350,   
            "cut_vertex_TrackAddChi2": 20,   
            "cut_vertex_TrackDropChi2": 6,       # NOT USED
            "cut_vertex_VertexChi2Reduced": 6,   
        }

    def run(self, tracks):
        self.tracks = tracks
        self.tracks_remaining = copy.copy(tracks)
        self.tracks_remaining_info = [] # a list of track infomation. Each element is [track_ind, track_chi2, track_vertex_chi2, track_vertex_dist]

        self.seeding(tracks)
        if self.debug: 
            for seed in self.seeds:
                print(seed)


        self.vertices = []
        while len(self.seeds)>0:
            # ------------------------------------
            # Round 1: Find tracks  that belongs to this vertex
            seed = self.seeds[-1]
            tracks_found, vertex_fit = self.find_once(self.tracks, seed)
            self.seeds.pop(-1)
            if len(tracks_found)==0:
                continue

            # ------------------------------------
            # Round 2: Drop outliers
            # tracks_found, vertex_found = self.fit_once(seed)

            # ------------------------------------
            # Round 3: Final fit
            # m = self.fit(tracks_found, vertex_location)


            # # Vertex = namedtuple("Vertex", ["x0", "y0", "z0", "t0", "cov", "chi2", "tracks"])
            vertex_location =  np.array(vertex_fit.values) 
            vertex_cov =  np.array(vertex_fit.covariance) 
            vertex_chi2 = vertex_fit.fval 
            vertex_tracks = tracks_found      
            self.vertices.append(datatypes.Vertex(*vertex_location, vertex_cov, vertex_chi2, vertex_tracks))

            # Finally, remove used tracks
            self.remove_related_seeds(tracks_found)

        return self.vertices

    def find_once(self, tracks, seed):
        # Add hits from the seed first
        if self.debug: print("--- New seed for vertex --- \n  Seed", seed)
        seed_inds = [seed.trackind1, seed.trackind2]
        seed_midpoint = np.array([seed.x0, seed.y0, seed.z0, seed.t0])
        tracks_found = [tracks[seed_inds[0]], tracks[seed_inds[1]]]
        # Fit the seed
        m = self.fit(tracks_found, seed_midpoint, hesse=False)
        if (not m.valid) or (m.fval>self.parameters["cut_vertex_SeedChi2"]):
            if self.debug: print(f"  * Seed failed. Seed fit result valid: {m.valid}, seed chi2 {m.fval}")
            return [], []        
        vertex_location = np.array(m.values)
        vertex_err = np.array(m.errors)
        vertex_chi2 = m.fval


        # Update the track info
        self.tracks_remaining_info = []
        for track in self.tracks_remaining:
            ind = track.ind
            chi2_track = track.chi2
            dist = Util.track.distance_to_point(track,vertex_location)
            chi2 = Util.track.chi2_point_track(vertex_location, track, point_unc=vertex_err)
            self.tracks_remaining_info.append(self.trackinfo(ind, chi2_track, chi2, dist))
        # Sort
        self.tracks_remaining_info.sort(key=lambda m: m.track_vertex_dist) # sort by distance


        # Add hits until no longer passes cut
        # iteratively add tracks     
        for i in range(len(self.tracks_remaining_info)):
            info = self.tracks_remaining_info[i]
            # Continue if track is from seed
            if info.track_ind in seed_inds:
                continue
            # Continue if track is too far from the seed
            if (info.track_vertex_chi2 > self.parameters["cut_vertex_TrackAddChi2"]) or\
               (info.track_vertex_dist > self.parameters["cut_vertex_TrackAddDist"]):
                if self.debug: print(f"  * Track [{info.track_ind}] too far from vertex. Track dist to vertex: {info.track_vertex_chi2:.2f}, track chi2 to vertex: {info.track_vertex_dist:.2f}")
                continue

            tracks_found.append(tracks[info.track_ind])
            m = self.fit(tracks_found, vertex_location, hesse=False)
            ndof = 3*len(tracks_found)-4
            if (not m.valid) or (m.fval/ndof>self.parameters["cut_vertex_VertexChi2Reduced"])\
                or ((m.fval-vertex_chi2)>self.parameters["cut_vertex_TrackAddChi2"]):
                if self.debug: print(f"  * Track [{info.track_ind}] removed from vertex fit. Fit valid: {m.valid}; vertex chi2_r {m.fval/ndof:.2f}; vertex chi2 increment {m.fval-vertex_chi2 :.2f}")                                   
                tracks_found.pop(-1)
                continue   
            vertex_location =  np.array(m.values) 
            vertex_chi2 = m.fval 
            if self.debug: print(f" -> Track [{info.track_ind}] added to vertex. Vertex chi2_r {m.fval/ndof:.2f}; vertex chi2 increment {m.fval-vertex_chi2 :.2f}. Track: {tracks[info.track_ind][:8]}") 



            # Update the track info  
            for j in range(i+1, len(self.tracks_remaining_info)):
                ind = self.tracks_remaining_info[j].track_ind
                track = tracks[ind]
                chi2_track = track.chi2
                dist = Util.track.distance_to_point(track,vertex_location)
                chi2 = Util.track.chi2_point_track(vertex_location, track, point_unc=vertex_err)
                self.tracks_remaining_info[j] = self.trackinfo(ind, chi2_track, chi2, dist)

        return tracks_found, m




    def remove_related_seeds(self, tracks_found):
        tracks_found_inds = [track.ind for track in tracks_found]
        tracks_found_inds.sort(reverse=True)
        if self.debug: print(f"Vertex found! track indices: {tracks_found_inds} \n")

        # Remove seeds 
        # Need to do backwards to not change the index
        for i in reversed(range(len(self.seeds))):
            seed = self.seeds[i]
            if (seed.trackind1 in tracks_found_inds) or (seed.trackind2 in tracks_found_inds):
                self.seeds.pop(i)  

        # Remove tracks
        # for i in reversed(range(len(self.tracks_remaining_info))):
        #     if self.tracks_remaining_info[i][0] in tracks_found_inds:
        #         self.tracks_remaining_info.pop(i)
        for i in reversed(range(len(self.tracks_remaining))):
            if self.tracks_remaining[i].ind in tracks_found_inds:
                self.tracks_remaining.pop(i)        
                   




    def seeding(self, tracks):
        seeds = []
        for i in range(len(tracks)):
            for j in range(i+1, len(tracks)):
                # Cut on seed distance
                midpoint,dist_seed = Util.track.closest_approach_midpoint_Track(tracks[i], tracks[j])
                if dist_seed>self.parameters["cut_vertex_SeedDist"]:
                    continue

                # # Cut on seed chi2
                # # This relys on a least square fit which is very slow 
                # m = self.fit([tracks[i],tracks[j]], midpoint, hesse=False)
                # if not m.valid or m.fval>self.parameters["cut_vertex_SeedChi2"]:
                #     continue
                # midpoint_fit = np.array(m.values)
                # midpoint_cov = np.array(m.covariance)
                # midpoint_chi2 = m.fval
                
                midpoint_chi2 = Util.track.chi2_point_track(midpoint, tracks[i])+ Util.track.chi2_point_track(midpoint, tracks[j])
                if midpoint_chi2>self.parameters["cut_vertex_SeedChi2"]:
                    continue
                

                # Check number of compatible tracks
                N_compatible_tracks = 0
                for k in range(len(tracks)):
                    if k in [i,j]:
                        continue
                    dist = Util.track.distance_to_point(tracks[k],midpoint)
                    N_compatible_tracks = N_compatible_tracks + 1 if dist<self.parameters["cut_vertex_TrackAddDist"] else N_compatible_tracks

                ## VertexSeed = namedtuple("VertexSeed",["x0", "y0", "z0", "t0", "cov", "chi2", "dist", "Ntracks", "trackind1","trackind2"])
                seeds.append(datatypes.VertexSeed(*midpoint, 0, midpoint_chi2, dist_seed, N_compatible_tracks, i, j))

        # Sort the seeds
        # Rank them with two keys
        # First the number of tracks that are compatible with this seed (seed.Ntracks, ascending)
        # Second the chi-square between two tracks of this seed (seed.chi2, descending)
        # The order is reversed because the seeds are used reversely. the last seed is  used first
        seeds.sort(key=lambda seed: (seed.Ntracks, -seed.chi2))
        self.seeds = seeds


    def fit(self, tracks, guess, hesse=False):
        x0_init, y0_init, z0_init,t0_init = guess

        m = iminuit.Minuit(chi2_vertex(tracks),x0=x0_init, y0=y0_init, z0=z0_init, t0=t0_init)
        m.limits["x0"]=(-100000,100000)
        m.limits["y0"]=(-100000,100000)
        m.limits["z0"]=(-100000,100000)
        m.limits["t0"]=(-100,1e5)
        m.errors["x0"]=0.1
        m.errors["y0"]=0.1
        m.errors["z0"]=0.1
        m.errors["t0"]=0.1

        m.migrad()  # run optimiser
        if m.valid and hesse:
            m.hesse()   # run covariance estimator
        return m



    
