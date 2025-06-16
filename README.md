# Deriving ENSO from Sea Surface Temperature Using PCA

This repository accompanies the blog post on using Principal Component Analysis (PCA), an undupervised learning method, to uncover dominant spatial and temporal patterns in sea surface temperature (SST), including the El Niño–Southern Oscillation (ENSO). The analysis demonstrates how mathematical decomposition alone can reveal complex and meaningful climate signals without prior labeling.

🔗 **Blog post**: [Deriving ENSO from SST using PCA](https://open.substack.com/pub/polarvertex/p/the-mathematical-structure-of-nature?r=394csb&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)  
📖 **Inspired by**: [kls2177/Climate-and-Geophysical-Data-Analysis](https://kls2177.github.io/Climate-and-Geophysical-Data-Analysis/chapters/Week7/Intro_to_PCA.html)

---

## Overview

The code in this repository:

- Downloads and preprocesses NOAA ERSSTv5 sea surface temperature data from 1854–2024.
- Applies PCA to extract dominant modes of SST variability across the equatorial Pacific.
- Visualizes the first three principal components as time series and spatial reconstructions.
- Compares the second PC to a derived Niño 3.4 index to show alignment with ENSO behavior.
- Animates SST projections onto the first three principal axes.
- Reconstructs the SST field using only the top three components to show dimensionality reduction.

---

## Contributions

Suggestions or contributions (especially around better mechanisms to interpret PC3!) are welcome. Feel free to open issues or PRs.

