import sys, os
import argparse
import importlib
import time



import kalmanfilter as KF
import vertexfinder as VF
import utilities as Util
import trackfinder as TF
import datatypes


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename',    type=str, help='Path: input filename')
    parser.add_argument('output_directory',  type=str, help='Path: output directory')
    parser.add_argument('--output_suffix',   type=str, default="", help='Path: (optional) suffix to the output filename')
    parser.add_argument('--io',     default="io_MuSim",  type=str, help='IO module to parse the input file. Default is io_MuSim in ./io_user/. Provide the full path if the IO file is not under ./io_user/')
    parser.add_argument('--config', default="",  type=str, help='Path: configuration file. Default configuration (config_defaut.py) will be used if no config file is provided.')
    parser.add_argument('--printn', default=1000,  type=int, help='Print every [printn] event')
    parser.add_argument('--debug',  action='store_true', help='Show debug info')
    args = parser.parse_args()

    # Initiate file IO
    current_dir = os.path.dirname(os.path.realpath(__file__))
    io_full_path = current_dir+ f"/io_user/{args.io}.py" if not os.path.exists(args.io) else args.io
    io_user = importlib.machinery.SourceFileLoader("*", io_full_path).load_module()


    # Parse the configuration
    config   = importlib.machinery.SourceFileLoader("*", "config_default.py").load_module()
    if len(args.config)>0:
        try:
            config_user = importlib.machinery.SourceFileLoader("*", args.config).load_module()
            for key in config_user:
                config[key]=config_user[key]
        except Exception as E:
            print("Error loading config file. Error:",E)

    # Initiate Track and Vertex finder
    tf = TF.TrackFinder(method="greedy", debug=(config.parameters["debug_tracker"] | config.parameters["debug"] | args.debug))
    vf = VF.VertexFitter(debug=(config.parameters["debug_vertexer"] | config.parameters["debug"] | args.debug))
    for key in config.parameters:
        tf.parameters[key] = config.parameters[key]
        vf.parameters[key] = config.parameters[key]


    #-------------------------------------------------------------
    # Load the file
    data = io_user.load(args.input_filename)
    
    # Make variables to hold the result
    results = {
        "hits":[],
        "tracks":[],
        "vertices":[],
    }  
    groups = list(data.keys())
    entries = len(data[groups[0]])
    entries_run = [config.parameters["start_event"], min(config.parameters["end_event"], entries)]

    # Some numbers for bookkeepping
    tracks_found=0
    tracks_found_events=0
    vertices_found=0
    vertices_found_events=0

    # Run track and vertex finding on all events
    print("Running...")
    time_start = time.time()
    for entry in range(*entries_run):
        if (entry+1)%config.parameters["print_n"]==0:  
            time_stop=time.time()
            print("    event is ", entry+1, ", time", time_stop-time_start, "seconds")
        results["hits"].append([])
        results["tracks"].append([])
        results["vertices"].append([])
        event_tracks=0
        event_vertices=0
        for group in groups:
            hits = data[group][entry]
            results["hits"][-1].extend(hits)
            if group!="inactive":
                tracks = tf.run(hits)
                vertices = vf.run(tracks) 
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
    print("Writing file to disk.")
    print("-------------------------")
    print("Summary")
    print("Events:",entries_run[1]-entries_run[0])
    print("Tracks:",tracks_found)
    print("Vertices:",vertices_found)
    print("Events with track:",tracks_found_events)
    print("Events with vertex:",vertices_found_events)
    print("-------------------------")

    # Save the results
    output_filename = os.path.abspath(args.output_directory) \
                        + "/"+ os.path.splitext(os.path.basename(args.input_filename))[0]\
                        + args.output_suffix
    io_user.dump(results, output_filename)
    print("Output saved as",output_filename)

            
if __name__ == "__main__":
    main()     