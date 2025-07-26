# ppg-basis
A python-based package to automate the decomposition photoplethysmographs into various basis functions for subsequent analysis and PPG reconstruction.

# Profiling
## Setup
- Install package locally with ```bash pip install .```
- To run profiling, run ```profile_full_pipeline.py``` to generate ```.prof``` files and open them
- To view ```.prof``` files, run ```view_profiles.py```

## Naming Convention
- ```*_pipeline.py``` - pipeline implementations
- ```testbench_*.py``` - profiling pipeline implementations