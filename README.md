# PPG-Basis

This Python-based package automates the following photoplethysmograph (PPG) functions:

* **Generation** of synthetic PPG signals
* **Decomposition** of PPGs into various basis functions
* **Reconstruction** of PPG signals using decomposed parameters

---

## Installation

### For Developers

1. **Clone** into your desired directory:

   ```bash
   git clone ppg-basis
   ```

2. **Navigate** into the project directory:

   ```bash
   cd ppg-basis
   ```

3. **Install** the package in editable mode with all dependencies from `setup.py`:

   ```bash
   pip install -e .
   ```

The package will now be available for import and use in your project.

### For Users

*TODO*

---

## Examples

* Refer to `base_tb.ipynb` for usage examples demonstrating the standard full functionality of the package.
* Refer to `custom_cost_func.ipynb` for examples of passing a custom cost function to the `ppgExtractor`.
* Refer to `_____.ipynb` for examples using experimental data to extract the AC/DC/HC components and reconstruct the AC component.

### Profiling

* To run profiling, execute:

  ```bash
  python profile_full_pipeline.py
  ```

  This will generate `.prof` files that can be opened for analysis.

* To view `.prof` files, run:

  ```bash
  python view_profiles.py
  ```

* File naming conventions:

  * `*_pipeline.py` — pipeline implementations
  * `testbench_*.py` — profiling pipeline implementations
