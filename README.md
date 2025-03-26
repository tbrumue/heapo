# HEAPO – An Open Dataset for Heat Pump Optimization with Smart Electricity Meter Data and On-Site Inspection Protocols

**Author:** Tobias Brudermüller (Brudermueller), Bits to Energy Lab, ETH Zurich: <tbrudermuell@ethz.ch>

This repository contains the Python code for the [following paper](https://arxiv.org/abs/2503.16993): 

> *Tobias Brudermueller, Elgar Fleisch, Marina González Vayá & Thorsten Staake (2025). HEAPO – An Open Dataset for Heat Pump Optimization with Smart Electricity Meter Data and On-Site Inspection Protocols.*

**Please note** that this work is currently under peer review and that the current manuscript on arxiv is a preprint. The dataset and dataloader are already available in their initial version, but updates may occur in future releases. If you use the dataset and or/related code in its initial form, you must **cite our work as follows**: 
```
@misc{brudermueller2025heapoopendataset,
title={HEAPO -- An Open Dataset for Heat Pump Optimization with Smart Electricity Meter Data and On-Site Inspection Protocols}, 
author={Tobias Brudermueller and Elgar Fleisch and Marina González Vayá and Thorsten Staake},
year={2025},
eprint={2503.16993},
archivePrefix={arXiv},
primaryClass={cs.CY},
url={https://arxiv.org/abs/2503.16993}, 
}
```

---

### Abstract 

Heat pumps are essential for decarbonizing residential heating but consume substantial electrical energy, impacting operational costs and grid demand. Many systems run inefficiently due to planning flaws, operational faults, or misconfigurations. While optimizing performance requires skilled professionals, labor shortages hinder large-scale interventions. However, digital tools and improved data availability create new service opportunities for energy efficiency, predictive maintenance, and demand-side management. To support research and practical solutions, we present an open-source dataset of electricity consumption from 1,408 households with heat pumps and smart electricity meters in the canton of Zurich, Switzerland, recorded at 15-minute and daily resolutions between 2018-11-03 and 2024-03-21. The dataset includes household metadata, weather data from 8 stations, and ground truth data from 410 field visit protocols collected by energy consultants during system optimizations. Additionally, the dataset includes a Python-based data loader to facilitate seamless data processing and exploration.

---

### Data  

The dataset itself is available on Zenodo and can be **downloaded here**: [Zenodo - HEAPO Dataset](https://zenodo.org/records/15056919).  

For a detailed explanation of the dataset structure, file contents, and parameters, please refer to the **dataset description paper**: [ArXiv - HEAPO Dataset Description](https://arxiv.org/abs/2503.16993). We strongly recommend reading this paper before working with the data.  

#### File Information  

- The dataset is provided as a compressed archive: **`heapo_data.zip`** (485 MB).  
- Once extracted, the dataset expands to **5.26 GB**.  

#### Usage Instructions  

To use the provided dataloader, you need to extract the dataset and specify its location (see example in `notebooks/01_usage_example.ipynb`):  

- You can store the extracted dataset at any location and specify its **absolute path** when initializing the dataloader object using the `data_path` argument (see the class initialization in `src/heapo.py`). 
- If `data_path=None`, the script assumes the dataset is located within this repository in a subfolder named **`heapo_data`**, alongside the `installation` and `heapo_data` directories.  
- **NOTE:** When initializing a dataloader object from the `HEAPO` class, it will automatically verify the dataset's completeness to ensure that all functionalities work correctly.

### Anaconda Installation  

If you prefer to use your own Python interpreter, ensure that all required packages listed in `installation/requirements.yml` are installed via `pip`.  

Alternatively, you can set up an Anaconda environment named **`heapo_env`** using the following steps:  

1. Navigate to the `installation` folder:  
   ```bash
   cd [path_to_repository]/installation
   ```  
2. Create the environment:  
   ```bash
   conda env create -n heapo_env -f requirements.yml
   ```  
3. Activate the environment:  
   ```bash
   conda activate heapo_env
   ```  
4. To exit the environment after your session:  
   ```bash
   conda deactivate
   ```

### Usage  

To get the most out of this repository and the HEAPO dataset, we recommend following these steps:  

1. Read the [data description paper](https://arxiv.org/abs/2503.16993).  
2. Explore some of the downloaded data files to familiarize yourself with the dataset.  
3. Review the example notebooks in the `notebooks` folder.  
4. Examine the dataloader implementation in `src/heapo.py` to understand its functionality and extend it as needed.  
5. Experiment with the dataloader and dataset to explore a use case of your interest.