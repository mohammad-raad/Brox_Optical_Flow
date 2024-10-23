# Brox Optical Flow Implementation

This repository provides a Python implementation of the Brox et al. optical flow algorithm. The code estimates dense optical flow between consecutive frames in image sequences, supporting multi-level processing, various linear solvers, and flow visualization.

Paper URL: https://www.mia.uni-saarland.de/Publications/brox-eccv04-of.pdf

Original paper: Brox, T., Bruhn, A., Papenberg, N., & Weickert, J. (2004). High accuracy optical flow estimation based on a theory for warping. In European conference on computer vision (pp. 25-36). Springer, Berlin, Heidelberg.


## Features

* **Multi-level approach:** Estimates flow at multiple image resolutions for improved accuracy and robustness.
* **Variational method:** Implements the robust variational optical flow formulation proposed by Brox et al.
* **Residual computation:** Uses fixed-point iteration to compute the residual.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mohammad-raad/Brox_Optical_Flow.git
   cd Brox_Optical_Flow
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
