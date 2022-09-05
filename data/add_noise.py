import numpy as np
from scipy.stats import poisson

#user defined modules


# Tight coefficients
NH_TIGHT = 0.3
LOGTEFF_TIGHT = 0.01
DIST_TIGHT = 0.05

# Loose coefficients
NH_LOOSE = 0.5
LOGTEFF_LOOSE = 0.025
DIST_LOOSE = 0.2

#scale
SCALE = 1

def add_noise_np(config,nH=[],logTeff=[],dist=[]):

    if len(nH) == 0:
        log_nH = np.random.normal(-1,1,size=config['n_stars'])
        nH = np.absolute(np.power(log_nH,10))
    if len(logTeff) == 0:
        logTeff = np.absolute(np.random.normal(6.15,0.05,size=config['n_stars']))
    if len(dist) == 0:
        dist = np.absolute(np.random.normal(7.3,2.0,size=config['n_stars']))

    #min, max of each NP
    nH_min = np.min(nH)
    nH_max = np.max(nH)

    logTeff_min = np.min(logTeff)
    logTeff_max = np.max(logTeff)

    dist_min = np.min(dist)
    dist_max = np.max(dist)

    # Generate numbers between -1 and 1 with the same shape as the nuisance_parameters
    random_sample = SCALE * np.random.uniform(-1, 1, size=[3,len(nH)])

    # Easiest way to think through this is to walk through the case where random_sample is all 1s
    # and again when random_sample is all -1s
    # In those cases you would be at the widths of the sampling ranges

    # Creating nH
    if config['nH'] == "tight":
        uncertainty = np.array([NH_TIGHT]) * random_sample[0]
        uncertainty += np.ones_like(uncertainty)
        nH = np.clip(nH * uncertainty,nH_min,nH_max)
    
    elif config['nH'] == "loose": 
        uncertainty = np.array([NH_LOOSE]) * random_sample[0]
        uncertainty += np.ones_like(uncertainty)
        nH = np.clip(nH * uncertainty,nH_min,nH_max)

    elif config['nH'] != "true":
        print("Invalid coefficent type for nH.")
        return

    # Creating logTeff
    if config['logTeff'] == "tight":
        uncertainty = np.array([LOGTEFF_TIGHT]) * random_sample[1]
        uncertainty += np.ones_like(uncertainty)
        logTeff = np.clip(logTeff * uncertainty,logTeff_min,logTeff_max)
    
    elif config['logTeff'] == "loose": 
        uncertainty = np.array([LOGTEFF_LOOSE]) * random_sample[1]
        uncertainty += np.ones_like(uncertainty)
        logTeff = np.clip(logTeff * uncertainty,logTeff_min,logTeff_max)

    elif config['logTeff'] != "true":
        print("Invalid coefficent type for logTeff.")
        return

    # Creating dist
    if config['dist'] == "tight":
        uncertainty = np.array([DIST_TIGHT]) * random_sample[2]
        uncertainty += np.ones_like(uncertainty)
        dist = np.clip(dist * uncertainty,dist_min,dist_max)
    
    elif config['dist'] == "loose": 
        uncertainty = np.array([DIST_LOOSE]) * random_sample[2]
        uncertainty += np.ones_like(uncertainty)
        dist = np.clip(nH * uncertainty,dist_min,dist_max)

    elif config['dist'] != "true":
        print("Invalid coefficent type for dist.")
        return

    return nH,logTeff,dist


def add_noise_spectrum(config, spectra):
    if(config['add_noise']):
        spectra = poisson.rvs(spectra,size=spectra.shape)
    return spectra
