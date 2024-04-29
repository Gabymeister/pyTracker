import sys, os
import argparse
import importlib
import time

import joblib
import pickle


from tracker import kalmanfilter as KF
from tracker import vertexfinder as VF
from tracker import utilities as Util
from tracker import trackfinder as TF
from tracker import datatypes
from tracker import config_default as config
import functools; print = functools.partial(print, flush=True) #make python actually flush the output!

def main():

    parser = argparse.ArgumentParser(     
                prog='pyTracker',
                description='Reconstructing track and vertex without magnetic field.',)
    parser.add_argument('input_filename',    type=str, help='Path: input filename')
    parser.add_argument('output_directory',  type=str, help='Path: output directory')
    parser.add_argument('--output_suffix',   type=str, default="", help='Path: (optional) suffix to the output filename')
    parser.add_argument('--io',     default="io_MuSim",  type=str, help='IO module to parse the input file. Default is io_MuSim in ./io_user/. Provide the full path if the IO file is not under ./io_user/')
    parser.add_argument('--config', default="",  type=str, help='Path: configuration file. Default configuration (config_defaut.py) will be used if no config file is provided.')
    parser.add_argument('--printn', default=1000,  type=int, help='Print every [printn] event')
    parser.add_argument('--debug',  action='store_true', help='Show debug info')
    parser.add_argument('--overwrite',  action='store_true', help='Overwrite the existing output file')
    args = parser.parse_args()

    # Initiate file IO
    current_dir = os.path.dirname(os.path.realpath(__file__)) # Path to this python file
    io_full_path = current_dir+ f"/io_user/{args.io}.py" if not os.path.exists(args.io) else args.io
    io_user = importlib.machinery.SourceFileLoader("*", io_full_path).load_module()
    output_filename = os.path.abspath(args.output_directory) \
                        + "/"+ os.path.splitext(os.path.basename(args.input_filename))[0]\
                        + args.output_suffix \
                        + ".joblib"    
    if os.path.exists(output_filename) and not args.overwrite:
        print("Output file exists. Processing terminated. Use --overwrite option to force running the tracker, or assign a different suffix by --output_suffix=.")
        return


    # Parse the configuration
    if len(args.config)>0:
        try:
            config_user = importlib.machinery.SourceFileLoader("*", args.config).load_module()
            for key in config_user.parameters:
                config.parameters[key]=config_user.parameters[key]
        except Exception as E:
            print("Error loading config file:",E)

    # Initiate Track and Vertex finder
    tf = TF.TrackFinder(method="greedy", debug=(config.parameters["debug_tracker"] | config.parameters["debug"] | args.debug))
    vf = VF.VertexFitter(debug=(config.parameters["debug_vertexer"] | config.parameters["debug"] | args.debug))
    for key in config.parameters:
        tf.parameters[key] = config.parameters[key]
        vf.parameters[key] = config.parameters[key]


    #-------------------------------------------------------------
    # Load the file
    data, metadata = io_user.load(args.input_filename, printn=2000, \
                                start_event=config.parameters["start_event"], end_event=config.parameters["end_event"])
    
    # Make variables to hold the result
    results = {
        "hits":[],
        "tracks":[],
        "vertices":[],
    }  
    groups = list(data.keys())
    entries = len(data[groups[0]])
    # entries_run = [config.parameters["start_event"], min(config.parameters["end_event"], entries)]

    # Some numbers for bookkeepping
    tracks_found=0
    tracks_found_events=0
    vertices_found=0
    vertices_found_events=0

    # Run track and vertex finding on all events
    print(f"Running on {entries} events...")
    time_start = time.time()
    for entry in range(entries):
        if (entry+1)%config.parameters["print_n"]==0 or args.debug:  
            time_stop=time.time()
            print("  Event is ", entry+config.parameters["start_event"], ", time", time_stop-time_start, "seconds")

        results["hits"].append([])
        results["tracks"].append([])
        results["vertices"].append([])
        event_tracks=0
        event_vertices=0
        
        for group in groups:
            hits = data[group][entry]
            results["hits"][-1].extend(hits)
            if group!="inactive":
                # Rotate hits so that y is always the layer direction
                if metadata["groups"][group]["flip_index"] is not None:
                    hits = [Util.general.flip_hit(hit, metadata["groups"][group]["flip_index"]) for hit in hits]
                    
                # Apply detector efficiency
                if 0<config.parameters["detector_efficiency"]<1:
                    hits = Util.processing.drop_hits(hits, config.parameters["detector_efficiency"], config.parameters["seed"])
                elif config.parameters["detector_efficiency"]!=1:
                    print("  Warning: detector efficiency is not in the range of (0,1]. Using default value 1.")
                    
                # Run track and vertex reconstruction
                tracks = tf.run(hits)
                vertices = vf.run(tracks) 

                # Rotate tracks and vertices back
                if metadata["groups"][group]["flip_index"] is not None:
                    tracks   = [Util.general.flip_track(track, metadata["groups"][group]["flip_index"]) for track in tracks]                
                    vertices = [Util.general.flip_vertex(vertex, metadata["groups"][group]["flip_index"]) for vertex in vertices]                

                # Save result
                results["tracks"][-1].extend(tracks)
                results["vertices"][-1].extend(vertices)
                tracks_found+=len(tracks)
                event_tracks+=len(tracks)
                vertices_found+=len(vertices)
                event_vertices+=len(vertices)

        tracks_found_events   += event_tracks>0
        vertices_found_events += event_vertices>0
    time_stop=time.time()
    print("Finished. Total time",time_stop-time_start, "seconds")
    print("-------------------------")
    print("Summary")
    print("  Events:",entries)
    print("  Tracks:",tracks_found)
    print("  Vertices:",vertices_found)
    print("  Events with track:",tracks_found_events)
    print("  Events with vertex:",vertices_found_events)
    print("-------------------------")

    # Save the results
    print("Writing file to disk.")
    joblib.dump(results, output_filename)
    print("Output saved as",output_filename)


            
if __name__ == "__main__":
    main()     