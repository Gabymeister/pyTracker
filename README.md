# Python package for track and vertex reconstruction

## Installation

### Local (Ubuntu)

Install cern ROOT, and then install python packages with a virtual environment.

```bash
cd ~
wget https://root.cern/download/root_v6.28.12.Linux-ubuntu22-x86_64-gcc11.4.tar.gz
tar xvzf root_v6.28.12.Linux-ubuntu22-x86_64-gcc11.4.tar.gz
echo "source ~/root/bin/thisroot.sh" >> ~/.bashrc

python -m venv venv_mathusla
echo "alias venv='source ~/venv_mathusla/bin/activate'" >> ~/.bashrc
venv
pip install numpy scipy matplotlib iminuit tqdm joblib scikit-learn uncertainties

git clone https://github.com/EdmondRen/pyTracker.git
cd pyTracker
pip install -e .
```

### Compute Canada

The SuperCDMS software release via singularity image has all the dependencies. The python package can be installed through the login shell of the singularity image. `--user` option is needed. 

```bash
module use --append /project/def-mdiamond/soft/modules
module load scdms/V05-00
singularity-shell
git clone https://github.com/EdmondRen/pyTracker.git
cd pyTracker
pip install -e .  --user
```

To run any program with the singularity image, use the follwing syntax (after running the module use and module load commands above):

```bash
singularity exec $SCDMS_IMAGE executable arguments...
```


## Usage

The main program is tracker/run.py, which will run the track and vertex finding and reconstruction. The main program can be run by passing the filename to python: 

    (singularity exec $SCDMS_IMAGE) python3 /path/to/tracker/run.py

The content in the bracket is for using CDMS singularity image.

Once installed, the program can be run directly with the format of

    (singularity exec $SCDMS_IMAGE) pytracker [-h] [--output_suffix OUTPUT_SUFFIX] [--io IO] [--config CONFIG] [--printn PRINTN] [--debug] [--overwrite] input_filename output_directory  
    
    positional arguments:
        input_filename        Path: input filename
        output_directory      Path: output directory

    options:
        -h, --help            show this help message and exit
        --output_suffix OUTPUT_SUFFIX
                                Path: (optional) suffix to the output filename
        --io IO               IO module to parse the input file. Default is io_MuSim in ./io_user/. Provide the full path if the IO file is not under ./io_user/
        --config CONFIG       Path: configuration file. Default configuration (config_defaut.py) will be used if no config file is provided.
        --printn PRINTN       Print every [printn] event
        --debug               Show debug info
        --overwrite           Overwrite the existing output file

Example:  
    
    (singularity exec $SCDMS_IMAGE) pytracker -h : this will show the help information  
    (singularity exec $SCDMS_IMAGE) pytracker INPUT_FILENAME OUTPUT_DIR

## Technical details


### Track reconstruction

The Kalman filter parameter names follows the convention of Fruhwirth paper (APPLICATION OF KALMAN FILTERING TO TRACK AND VERTEX FITTING).

### Vertex reconstruction

