# Investigating The Allegation of Soil Pollution in the Kalumbila Mineralized Zone

## Project Overview

This repository contains the research concept and methodological framework for investigating allegations of **soil pollution in the Kalumbila Mineralized Zone**, located in Northwestern Zambia. The project introduces an **integrated data science and geostatistical approach** to objectively differentiate **natural geochemical enrichment** from **potential anthropogenic contamination** in a highly mineralized environment.

All analyses, workflows, and exploratory notes are documented in [`concept_notes.ipynb`](./concept_notes.ipynb).

---

## Research Problem & Motivation

Mineralized areas such as Kalumbila pose a unique environmental challenge: natural geological processes produce elevated metal concentrations, which may overlap with contamination from mining activities. Traditional environmental assessment methods, which rely on **single-element regulatory thresholds**, often fall short because they:

- Ignore **multivariate interactions** between geochemical elements  
- Cannot clearly distinguish **natural enrichment from anthropogenic pollution**  
- Overlook **spatial context and autocorrelation**  
- Fail to quantify **uncertainty in interpretation**

This project addresses these limitations by developing a **data science and machine learning methods framework**.

---

## Methodological Framework

### Core Analytical Pipeline

1. **Compositional Data Analysis (CoDA)**
   - **Purpose:** Correctly handle closed geochemical data (percentages, ppm) to avoid spurious correlations.  
   - **Methods:** Log-ratio transformations (ilr/clr), ternary diagrams, variation arrays, biplots.

2. **Multivariate Outlier Detection**
   - **Purpose:** Identify samples with anomalous *combinations* of elements rather than single-element extremes.  
   - **Methods:** Robust Mahalanobis distances (MCD), Chi-square Q-Q plots, sensitivity analysis.

3. **Pattern Recognition & Dimensionality Reduction**
   - **Purpose:** Reveal dominant geochemical processes and reduce dataset complexity.  
   - **Methods:** PCA on ilr-coordinates, scree plots, and interpretation of loadings.

4. **Spatial Analysis & Modeling**
   - **Purpose:** Quantify and model the spatial structure of contaminants.  
   - **Methods:** Variogram analysis, anisotropy assessment, kriging interpolation.

5. **Probabilistic Source Attribution**
   - **Purpose:** Move beyond deterministic classification to quantify uncertainty in contamination identification.  
   - **Methods:** Bayesian classification with continuous predictors, posterior probability mapping.

---

### Supporting & Specialized Protocols

- **Extreme Value Analysis (EVA):** For modeling tail risks and extreme contaminant concentrations.  
- **Clustering Analysis:** For empirical definition of background geochemical populations.  
- **Kernel Density Estimation (KDE):** Non-parametric estimation of probability densities for Bayesian models.

---



