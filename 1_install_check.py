# 1_install_check.py
# Purpose: Verify all installations are working correctly

import sys
import ray
import sklearn
import pandas
import numpy
import matplotlib

print("=" * 50)
print("SETUP VERIFICATION")
print("=" * 50)
print(f"Python Version     : {sys.version}")
print(f"Ray Version        : {ray.__version__}")
print(f"Scikit-learn       : {sklearn.__version__}")
print(f"Pandas             : {pandas.__version__}")
print(f"NumPy              : {numpy.__version__}")
print(f"Matplotlib         : {matplotlib.__version__}")
print("=" * 50)
print("All packages loaded successfully!")