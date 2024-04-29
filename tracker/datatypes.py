from collections import namedtuple

Hit = namedtuple("Hit", ["x", "y", "z", "t", "x_err", "y_err", "z_err", "t_err","layer", "ind"])
Track = namedtuple("Track", ["x0", "y0", "z0", "t0", "Ax", "Ay", "Az", "At", "cov", "chi2", "ind", "hits", "hits_filtered"])
Vertex = namedtuple("Vertex", ["x0", "y0", "z0", "t0", "cov", "chi2", "tracks"])
VertexSeed = namedtuple("VertexSeed",["x0", "y0", "z0", "t0",  "cov", "chi2","dist", "Ntracks", "trackind1","trackind2", "score"])