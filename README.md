# SUBPREP: SUBcellular calcium imaging PREProcessing toolbox

SUBPREP is a python toolbox specifically designed for subcellular calcium imaging. It is a computationally efficient preprocessing pipeline for ROI selection (after using video preprocessing softwares like suite2p), movement artifact identification, and ROI grouping for axon imaging.
For more detailed description, please go to [bioRxiv link](https://www.biorxiv.org/content/10.1101/2024.10.04.616737v1.)

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1. Download the Repository](#1-download-the-repository)
  - [2. Set Up a Virtual Environment Using `env.yml`](#2-set-up-a-virtual-environment-using-envyml)
  - [3. Install the SUBPREP Package](#3-install-the-subprep-package)
- [Demo](#demo)
- [Contact](#troubleshooting)

## Prerequisites

Before you begin, ensure you have the following installed:
- [Python 3.x](https://www.python.org/downloads/): SUBPREP requires Python version 3.x to run.
- [Git](https://git-scm.com/): Git is required to download the repository.
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/): Conda is recommended for managing the virtual environment.

## Installation

### 1. Download the Repository
You can either download SUBPREP directly as a ZIP file or use Git to clone the repository.

#### Option 1: Download as ZIP
1. Go to the [SUBPREP GitHub page](https://github.com/anqijiang/subprep).
2. Click the green **Code** button and select **Download ZIP**.
3. Extract the ZIP file to a location of your choice.

#### Option 2: Clone via Git
If you have Git installed, you can clone the repository directly:
1. Open a terminal (eg. Anaconda prompt).
2. Run the following command:
   ```bash
   git clone https://github.com/anqijiang/subprep.git

### 2. Set up a virtual environment using `env.yml`
1. Open a terminal (eg. Anaconda prompt).
2. Navigate to the SUBPREP folder from last step
3. Create the virtual environment using the following command: (this step will take a while)
   ```bash
   conda env create -f env.yml
4. Activate the virtual environment:
   ```bash
   conda activate subprep

### 3. Install the SUBPREP package
1. Under the package directory and subprep virtual environment activated, run:
   ```bash
   pip install -e .
   ```
    This step will allow the package to be editable. You can adjust the code based on your research needs.

## Demo
Once the environment is set up and subprep installed, you can begin playing around using `demo.ipynb` under `jupyter` folder.
1. Ensure that the virtual environment is activated.
2. Open the jupyter notebook file.
3. Change the directory in the jupyter notebook file to your local directory to load example data.

## Contact
If you have any questions or need additional assistance, feel free to contact: [anqijiang@uchicago.edu](mailto:anqijiang@uchicago.edu)
Any feedbacks welcomed.