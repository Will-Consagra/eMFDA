# Overview

Code implementing the multidimensional functional data representation methods from "Efficient Multidimensional Functional Data Analysis Using Marginal Product Basis Systems" (https://arxiv.org/pdf/2107.14728.pdf)

# Computing

## Set up python environment 

conda create --name eMFDApy38 python=3.8
conda activate eMFDApy38
conda install -c conda-forge scikit-fda
conda install -c anaconda jupyter 

conda install -c tensorly tensorly==0.5.1
conda install seaborn

conda activate eMFDApy38

## Set up R environment

Install R version 4.2.2

Open R and install the following libraries

1. reticulate
2. hero

## Getting the Code

git clone https://github.com/Will-Consagra/eMFDA.git

# Simulation Analysis

A) To recreate the subset of the simulation study reported in Figure 1 of Section 5.1 of the main text, please run the following commands:

1. simulation_51_fig1.py --DATA_DIR /path/2/store/data/ --Nreps 100

The results will be written to /path/2/store/data/results/margarita_fcptpa_mise_comparison.png

NOTE: Due to the number of comparisons being made, full simulation can take a long time (~ 1.5 days on 2.4 GHz Intel Xeon CPU E5-2695 and 24GB of RAM)

B) To recreate the results reported in Table 1 of Section 5.1 of the main text, please perfom the following steps:

1. For K_t = 20 and both sigma2_e = 10.0, 0.5 do 2-3:

2. python simulation_51_hyper_param.py --MODE simulate --DATA_DIR /path/2/store/data/ --Nreps 100 --Kt K_t --sigma2_e sigma2_e

3. python simulation_51_hyper_param.py --MODE analyze --DATA_DIR /path/2/store/data/ --Nreps 100 --Kt K_t --sigma2_e sigma2_e

4. For K_t = 20, do: Rscript simulation_51_hyper_param.R /path/2/store/data/ 100 K_t

5. python compile_simulation_analysis_51_hyperparam.py --DATA_DIR /path/2/store/data/

C) To recreate the simulation study in Section 5.2, simply run the Ipython notebook simulation_52.ipynb

D) To recreate the automated rank selection Figure S1 and S2, simply run the Ipython notebooks rank_selection_evaluation_51.ipynb (for set-up from Section 5.1) and additional_evaluation_52.ipynb (for set-up from 5.2)

E) To recreate the simulation study in S5.1 of the supplement, please perform the following steps.

1. python simulation_S51.py --MODE simulate --DATA_DIR /path/2/store/data/ --Nreps 100 
2. for i in $(seq 1 1 Nreps); do python simulation_S51.py --MODE analyze --DATA_DIR /path/2/store/data/ --rep $i; done; 
3. for i in $(seq 1 1 Nreps); do Rscript simulation_S51.R /path/2/store/data/ $i; done;
4. python compile_simulation_analysis_S51.py --DATA_DIR /path/2/store/data/

Figures + csv files summarizing the analysis will be written to /path/2/store/data/results

WARNING: This final simulation study produces an enormous amount of data (approximately 1TB).
