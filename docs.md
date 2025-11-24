# Project Documentation

## Introduction
The characterization of exoplanetary atmospheres allows a deeper understanding of planetary formation,
evolution, and habitability through atmospheric retrieval. The growing interest in these topics is confirmed
by the expected launch in 2029 of the Atmospheric Remote-sensing Infrared Exoplanet Large-survey
(ARIEL) space mission by the European Space Agency, which will conduct a four-year large-scale
spectral survey of transiting exoplanets at various levels of detail. Atmospheric retrieval refers to the
fundamental process of inferring the chemical properties of the exoplanetary atmospheres. During the
last two decades, a plethora of exoplanet atmospheric retrieval codes have been developed, demonstrating
the large interest in this research topic. Currently, the state-of-the-art atmospheric retrieval methods are
based on Bayesian inference. However, in certain cases these methods require several weeks to
produce a full posterior distribution of atmospheric parameters given observed spectra. With the expected
volume of incoming observations, traditional methods become simply impractical. Furthermore, the
scalability requirement for atmospheric retrieval methods is supported by the need of high-resolution
spectroscopic observations for accurate retrievals. Thanks to the increased sensitivity of high-resolution
instruments, spectral bands can be decomposed into a dense forest of individual spectral lines, enabling
precise molecular identification at the cost of significant computational burden. To address these
challenges, we propose a novel atmospheric retrieval framework based on Continuous Normalizing Flows
(CNFs) trained via Optimal-Transport Conditional Flow Matching (OT-CFM) to compute the
joint posterior distribution of atmospheric parameters.

## Methods
CNFs are a class of deep generative models that learn the probability flow from simple, easy-to-sample
distributions to complex, high-dimensional probability distributions. Recently, OT-CFM has been
introduced as a simulation-free training paradigm based on regressing vector fields of fixed conditional
probability paths, enabling for scalable training and sampling. Unlike conventional neural networks that
produce point estimates, CNFs can approximate the full posterior distribution of atmospheric parameters
given observed spectra by solving the corresponding ODE formulation through multiple neural function
evaluations (NFEs). Compared to NS, our CNF-based retrieval framework offers several advantages. First,
it significantly accelerates inference as the time needed for multiple NFEs to solve the ODE formulation is
significantly less than the time required by sampling methods to produce a posterior sample; once trained,
the model can generate a full posterior distribution in few hours, making it ideal for large-scale exoplanet
characterization studies. Second, CNFs enables not only the generation of posterior samples but also the
computation of the corresponding log-probabilities. Third, the flexibility of CNFs allows for seamless
adaptation to different observational setups (e.g. low/high-resolution, ground/space-based instruments,
etc.). To the same extent of traditional retrieval methods, CNFs could provide a robust uncertainty
quantification. We plan to compute the full posterior distributions of atmospheric parameters using CNFs
and to evaluate them in terms of calibration and uncertainty quantification, allowing a fair comparison
between different statistical models. Our work will leverage NVIDIA’s GPU-accelerated libraries, such as
CUDA and cuDNN, in Python programming language, and cloud-hosted NVIDIA’s hardware.

## Results


## Reproducibility
To reproduce our results:
```bash
git clone https://github.com/your-username/repo-name.git
```
