parameters = {}


# Run parameters
parameters["debug"]=False
parameters["debug_tracker"]=False
parameters["debug_vertexer"]=False
parameters["print_n"]=100
parameters["start_event"]=0
parameters["end_event"]=10000

# Global parameters:
parameters["multiple_scattering_p"]: 500 # [MeV/c] momentum of multiple scattering, 
parameters["multiple_scattering_length"]: 0.06823501107481977 # [1] material thickness in the unit of scattering length
# Example for calculating the multiple_scattering_length
# L_Al =  0.4 # [cm]
# L_Sc = 1.0 # [cm] Scintillator
# L_r_Al = 24.0111/2.7; # [cm] Radiation length Aluminum/ density of Aluminum
# L_r_Sc = 43; # [cm] Radiation length Scintillator (Saint-Gobain paper)
# multiple_scattering_length = L_Al / L_r_Al + L_Sc / L_r_Sc; # [rad lengths] orthogonal to Layer

# Track parameters
parameters["cut_track_HitAddChi2"]=12
parameters["cut_track_HitDropChi2"]=7
parameters["cut_track_HitProjectionSigma"]=7
parameters["cut_track_TrackChi2Reduced"]=3
parameters["cut_track_TrackChi2Prob"]=0.9
parameters["cut_track_TrackNHitsMin"]=3
parameters["cut_track_TrackSpeed"] = [20,35]
parameters["fit_track_MultipleScattering"]=True
parameters["cut_track_MultipleScatteringFind"]=False
parameters["fit_track_Method"]="backward"  #"least-square"

# Vertex parameters
parameters["cut_vertex_SeedDist"]=200
parameters["cut_vertex_SeedChi2"]=100
parameters["cut_vertex_TrackAddDist"]=200
parameters["cut_vertex_TrackAddChi2"]=25
parameters["cut_vertex_TrackDropChi2"]=15
parameters["cut_vertex_VertexChi2ReducedAdd"]=10
parameters["cut_vertex_VertexChi2Reduced"]=7