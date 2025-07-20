#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ppg_basis import ppgGenerator
from ppg_basis import ppgExtractor
import matplotlib.pyplot as plt
import time

# In[ ]:

start = time.perf_counter()

# Generate signal
ppgGen = ppgGenerator(fs=125,
                      hr=60,
                      mu=1,
                      sigma=0,
                      duration=10,
                      L=4,
                      basis_type="gamma")

sig = ppgGen.generate_signal()

end = time.perf_counter()

print(f"ppgGenerator with parameters Round 1 Runtime: {end - start:.4f} seconds")

start = time.perf_counter()

# Generate signal
ppgGen = ppgGenerator(fs=125,
                      hr=60,
                      mu=1,
                      sigma=0,
                      duration=10,
                      L=4,
                      basis_type="gamma")

sig = ppgGen.generate_signal()

end = time.perf_counter()

print(f"ppgGenerator with parameters Round 2 Runtime: {end - start:.4f} seconds")

# In[ ]:

start = time.perf_counter()

# Extract signal parameters
ppgExt = ppgExtractor(signal=sig,
                      fs=125,
                      hr=60,
                      sigma=0,
                      L=4,
                      basis_type='gamma')

theta_pred, params_pred = ppgExt.extract_ppg()

end = time.perf_counter()

print(f"ppgExtractor Round 1 Runtime: {end - start:.4f} seconds")

start = time.perf_counter()

# Extract signal parameters
ppgExt = ppgExtractor(signal=sig,
                      fs=125,
                      hr=60,
                      sigma=0,
                      L=4,
                      basis_type='gamma')

theta_pred, params_pred = ppgExt.extract_ppg()

end = time.perf_counter()

print(f"ppgExtractor Round 2 Runtime: {end - start:.4f} seconds")



# In[ ]:

start = time.perf_counter()

# Generate PPG using extracted parameters
ppgPrd = ppgGenerator(fs=125,
                      hr=60,
                      mu=1,
                      sigma=0,
                      duration=10,
                      L=4,
                      basis_type="gamma",
                      thetas=theta_pred,
                      params=params_pred)
pred = ppgPrd.generate_signal()

end = time.perf_counter()

print(f"ppgGenerator with extracted parameters Runtime: {end - start:.4f} seconds")

start = time.perf_counter()

# Generate PPG using extracted parameters
ppgPrd = ppgGenerator(fs=125,
                      hr=60,
                      mu=1,
                      sigma=0,
                      duration=10,
                      L=4,
                      basis_type="gamma",
                      thetas=theta_pred,
                      params=params_pred)
pred = ppgPrd.generate_signal()

end = time.perf_counter()

print(f"ppgGenerator with extracted parameters Round 2 Runtime: {end - start:.4f} seconds")


# In[5]:


# Fig. 4: Show utility in phase extraction and data augmentation
# Later part of paper could argue about augmentated trial datasets, and use for future ML
# Noise resistance models w/ DNNs
