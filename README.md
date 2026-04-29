# Benchmarking Variational Quantum Classifiers Against Classical SVCS with US Unemployment Rate

Our full project writeup is included [here.](./Writeup.pdf)

For our project we aim to train quantum machine learning problems to see how they compare to normal prediction models (ie sckit regression models) for nonlinear datasets. Our initial test will be conducted on the [Federal Reserve Bank of ST. Louis's FRED](https://fred.stlouisfed.org/graph/?m=1wm8j#) unemployment rate statistics from January 1st 2020 to January 1st 2026. 

### Dataset Description (FRED unemployment)
FRED's Unemployment Rate information is essentially a month by month catalog of the percent rate of unemployment in US cities and towns (designated as metropolitan areas). The data is collected via the US Bureau of Labor Statistics. The percentage is from 0% to 100%, if data was not collected, it was left as empty or NULL.

Cited from U.S. Bureau of Labor Statistics, Unemployment Rate in St. Louis, MO-IL (MSA) [STLUR], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/STLUR, April 22, 2026. 

### Project Hypothesis:
We predict that, under current quantum machine learning capabilities, the classical SVM model will achieve higher accuracy than the quantum classifier on both the Two Moons dataset (our testing dataset) and also on the US unemployment dataset (real-world data). This is primarily due to severe limitations of quantum hardware and simulation, which hinder the quantum model's ability to detect nonlinear decision boundaries. By directly comparing the two models, we can assess the present limits of quantum machine learning in simple nonlinear tasks, using the Two Moons benchmark.

**The figure below** shows a pair plot of unemployment features. The display includes the distribution and pairwise relationships for the 4-month sliding window features.

<img src="./Quantum Figure 1.png">
