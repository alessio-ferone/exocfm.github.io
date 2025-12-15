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
CNFs are a class of deep generative models that learn the probability flow from simple, easy-to-sample distributions to complex, high-dimensional probability distributions. Recently, OT-CFM has been introduced as a simulation-free training paradigm based on regressing vector fields of fixed conditional probability paths, enabling for scalable training and sampling. Unlike conventional neural networks that produce point estimates, CNFs can approximate the full posterior distribution of atmospheric parameters
given observed spectra by solving the corresponding ODE formulation through multiple neural function evaluations (NFEs). Compared to NS, our CNF-based retrieval framework offers several advantages. First, it significantly accelerates inference as the time needed for multiple NFEs to solve the ODE formulation is significantly less than the time required by sampling methods to produce a posterior sample; once trained, the model can generate a full posterior distribution in few hours, making it ideal for large-scale exoplanet
characterization studies. Second, CNFs enables not only the generation of posterior samples but also the computation of the corresponding log-probabilities. Third, the flexibility of CNFs allows for seamless adaptation to different observational setups (e.g. low/high-resolution, ground/space-based instruments, etc.). To the same extent of traditional retrieval methods, CNFs could provide a robust uncertainty quantification. We plan to compute the full posterior distributions of atmospheric parameters using CNFs
and to evaluate them in terms of calibration and uncertainty quantification, allowing a fair comparison between different statistical models. Our work will leverage NVIDIA’s GPU-accelerated libraries, such as CUDA and cuDNN, in Python programming language, and cloud-hosted NVIDIA’s hardware.

## Data
In collaboration with the Italian National Institute of Astrophysics (INAF), we build a dedicated high-resolution dataset following the structure of the Ariel Data Challenge 2022 / 2023 datasets. In particular, the dataset comprises $91,392$ samples, each including three primary components:
* Spectral data, comprising of a 102,400-dimensional atmospheric spectrum, providing information on transit depth and covering the spectral range from 0.9 $\mathrm{\mu m}$ to 2.42 $\mathrm{\mu m}$ and associated measurement uncertainty.
* Auxiliary data, encompassing 9 additional stellar and planetary parameters, such as star distance, stellar mass, stellar radius, stellar temperature, planet mass, planet radius, orbital period, semi-major axis, and surface gravity.
* Target data, describing 7 atmospheric parameters: the planet radius (in Jupyter radii $\mathrm{R_J}$), temperature (in Kelvin), and the log-abundance of five atmospheric gases such as $( \text{H}_2\text{O}$) (water), 4( \text{CO}_2 $) (carbon dioxide), $(\text{CO}$) (carbon monoxide), $( \text{CH}_4 $) (methane), and $( \text{NH}_3 $) (ammonia gas). These are the input parameters generating the simulated observations.
We split the samples into training, validation, and test sets according to the common 70 (63,973 samples) / 20 (18,278 samples) / 10 (9,140 samples) ratio.

To avoid finite-precision numerical issues, the stellar radius and mass are expressed in units of Solar radii and masses, while the planet radius and mass are expressed in units of Jupyter radii and masses. 
Then, stellar and planetary temperatures are transformed in log space to be consistent with the rest of the atmospheric parameters.
Due to the very narrow observed transit depth range ($10^{-5}-10^{-2}$), trasmission spectra and measured uncertainties are scaled by a factor of $10^{3}$.
As required by the flow matching paradigm, target atmospheric parameters and auxiliary data are preprocessed using feature-wise Z-score normalization according to the mean and standard deviation computed on samples of the training set.
Finally, we naturally perform online data augmentation by perturbing each transmission spectrum with Gaussian noise having zero mean and variance given by the square of the associated spectral uncertainties

## Experimental setup

Therefore, we compute the prediction errors between the input parameters and posterior samples, and between the ideal/real spectra and median predictive spectra, to measure the predictive performance of a probabilistic estimator. Median predictive spectra are obtained by passing the posterior distribution of atmospheric parameters as input to the forward model. The common metrics for prediction error include \emph{Mean Absolute Error} (MAE), \emph{Mean Squared Error} (MSE), \textcolor{blue}{\emph{Median Absolute Error} (MedAE),} and \emph{Root Mean Squared Error} (RMSE). 
These metrics alone only partially describes the predictive performance of a probabilistic estimator. 
Therefore, a qualitative and quantitative assessment of the uncertainty and calibration must be considered to check: (i) how a given estimator is confident about its predictions; and (2) whether the predicted values match the empirical frequencies of observed values. 
These tests provide precious information about the quality of the predicted posterior distributions but to check whether the inference is correct, the analysis should be complemented with posterior predictive checks (PPCs; \cite{cook_validation_2006, gelman_bayesian_2013}). These checks involve the comparison between the distribution of the original simulated observations with the posterior predictive distribution obtained by passing the set of posterior samples as input to the simulator. 
Due to the slow sampling speed of our simulator, PPCs cannot be performed in a reasonable amount of time, even considering the modest size of the designed test set.
To partially overcome this computational challenge, posterior coverage analysis provides valuable information about the soundness of the estimated posterior distributions.
The following sections describe rationales, criteria, and metrics behind each stage of the proposed posterior evaluation framework. 

## Results
We perform a comparative analysis including our method, the FMPE baseline, and NPE. We plan to extend the comparison to traditional Bayesian inference method (e.g. DE-MCMC) on a very limited subset of the test set, due to their significant computational cost.
To assess the performance of heterogeneous posterior estimators, we established an extensive evaluation framework.


### Regression Errors
Prediction errors between the target parameters and posterior samples are measured using emph{Mean Absolute Error} (MAE), \emph{Mean Squared Error} (MSE), \textcolor{blue}{\emph{Median Absolute Error} (MedAE),} and \emph{Root Mean Squared Error} (RMSE). 


- Table
- Discussion

### Uncertainty Estimation and Calibration
Calibration evaluates whether the distribution of predicted values obtained from a model matches empirical frequencies. For example, a 95% credible interval should contain the true value 95% of the time. Well-calibrated uncertainties are crucial for trustworthy decision-making. 
To measure the uncertainty and calibration of the predictions produced by a given probabilistic regression model, we considered the following metrics: Negative Log-Likelihood (NLL), Pinball Loss, Quantile Calibration Error (QCE), Uncertainty Calibration Error (UCE), and Expected Normalized Calibration Error (ENCE) are considered.
We perform a qualitative assessment of the calibration associated to the predictions of a given probabilistic regression model through a \emph{Reliability Regression} diagram (also known as \emph{reliability plot}), which visually compares the predicted and the observed quantile coverage. 

- Table / Figure?
- Discussion

### Posterior Coverage
Posterior Coverage Analysis is a diagnostic tool used to quantitatively evaluate the ground-truth benchmarking performance of a probabilistic regression method. 
The basic idea is that the predicted posterior distribution of a probabilistic regression model should at least include the true input parameters passed as input to the simulator to generate the corresponding observation. To this aim, we define two posterior coverage metrics: the Marginal Coverage Ratio (MCR) and the Joint Coverage Ratio (JCR). For a given confidence interval, the former measures the average fraction of values of the true atmospheric parameters falling within the sets of marginal posterior values, while the latter measures the average fraction of the true atmospheric parameters for which the values along each dimension fall jointly within the sets of marginal posterior values.

- Table / Figure Corner Plot
- Discussion


### Divergence
We complete our posterior evaluation framework by including metrics for quantifying the differences between probability distributions. In this case, we considered the Maximum Mean Discrepancy (MMD) and the Jensen-Shannon Divergence (JSD).
Lower values of both metrics indicate more similarity between distributions.

- Table


## Reproducibility
To reproduce our results:
```bash
git clone https://github.com/gomax22/fm4ar.git
```


## References?
