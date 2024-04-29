# Track finding with Kalman filter


The Kalman filter parameter names follows the convention of Fruhwirth paper (APPLICATION OF KALMAN FILTERING TO TRACK AND VERTEX FITTING).


TODO:
1. Tracker: Add multiple scattering matrix calculation 
2. Tracker: Add seed ranking

## Install

### Dependencies

```bash
cd ~
wget https://root.cern/download/root_v6.28.12.Linux-ubuntu22-x86_64-gcc11.4.tar.gz
tar xvzf root_v6.28.12.Linux-ubuntu22-x86_64-gcc11.4.tar.gz
echo "source ~/root/bin/thisroot.sh" >> ~/.bashrc

python -m venv venv_mathusla
echo "alias venv='source ~/venv_mathusla/bin/activate'" >> ~/.bashrc
pip install pyqt6 opencv-python-headless uproot numpy scipy matplotlib ipython jupyter iminuit tqdm joblib scikit-learn uncertainties h5py
```

## 
pip install -e . --user
