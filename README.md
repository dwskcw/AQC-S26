# AQC-S26

## Objective:
For our project we aim to train quantum machine learning problems to see how they compare to normal prediction models (ie sckit regression models) for nonlinear datasets. Our initial test will be conducted on the [Make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) data set and then the [Federal Reserve Bank of ST. Louis's FRED](https://fred.stlouisfed.org/graph/?m=1wm8j#) unemployment rate statistics from January 1st 2020 to January 1st 2026. 

## Datasets Description:
Make_moons is a (by default) 100-sample dataset included as part of Python’s scikit-learn machine learning library. When plotted, the samples produce two interleaving half-circles across a horizontal range of -1.0 to 2.0 and vertical range of -0.5 to 1.0. The library provides options to add Gaussian noise to the samples, which will randomly scatter the samples by a set amount, and additionally we can shuffle the order of the samples.

FRED's Unemployment Rate information is essentially a month by month catalog of the percent rate of unemployment in US cities and towns (designated as metropolitain areas). The data is collected via the US Bureau of Labor Statistics. The percentage is from 0% to 100%, if data was not collected, it was left as empty or NULL.

Cited from U.S. Bureau of Labor Statistics, Unemployment Rate in St. Louis, MO-IL (MSA) [STLUR], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/STLUR, April 22, 2026. 

## Project Hypothesis:
We predict that, under current quantum machine learning capabilities, the classical SVM model will achieve higher accuracy than the quantum classifier on both the Two Moons dataset (our testing dataset) and also on the US unemployment dataset (real-world data). This is primarily due to severe limitations of quantum hardware and simulation, which hinder the quantum model's ability to detect nonlinear decision boundaries. By directly comparing the two models, we can assess the present limits of quantum machine learning in simple nonlinear tasks, using the Two Moons benchmark.

## QML method (Part 1):
- To test our hypothesis that classical Support Vector Machines (SVM) models outperform quantum classifiers, we will implement our own Quantum Machine Learning (QML) classifier using a quantum kernel. This will initially test how well we should expect quantum should be in comparison and how it deals with fake non-linear data.

1. Preparing Data: Since our dataset will be the Two Moons dataset, our data consists of the coordinates of each point and a binary label of which moon it belongs to (0 = bottom moon, 1 = top moon).  To begin, we will split our data so 70% will be the training set and 30% belong to the testing set, and we will normalize the coordinates.
   
2. Quantum Feature Encoding: Using the ZZ feature map, we will map our normalized classical data points into a quantum state, effectively transforming our 2d points into a quantum Hilbert Space.

3. Kernel Approach: Next, we will compute a quantum kernel matrix to measure the similarity of points in the quantum feature space. Since we have a few features and parameters, training shouldn’t be an issue.

4. Classical Optimization: Once we compute the kernel matrix, we need to send it to a classical SVM. Our SVM uses the kernel matrix to learn an optimal decision boundary between our two moons. This must be done because quantum computing hardware is not yet capable of solving convex optimization models in a timely or reliable manner.

5. Evaluations & Comparison: After the training, we will test our model on our training set. Accuracy will be our metric of success. To compare results, we will also train a classical SVM with an RBF kernel on the same dataset and compare classification accuracy.

6. Final Report: Based on our findings, we will be able to assess our quantum kernel classifier on the moons dataset compared to a classical SVM. Through this process, we will justify or disprove our hypothesis. To conclude our analysis, we will discuss the main sources of error in the quantum model, which will include noise, limited qubit encoding, and possible issues with our feature map.

## QML method (Part 2):
Once we have a grasp of how the models should look like, we will modify our approach to work with unemployment prediction, which is inherently non-linear but is a good example of a real-world problem that could benefit from quantum computing

1. Setup/Preparing Data: Since we have already worked with classification in makeMoons, we can convert our data to a classification problem by having two options by using 1 to indicate rising unemployment, and 0 represent decreasing unemployment. Using 4-month periods we can group our data so that using the last 4 months we can attempt to predict unemployment rates changing.

2. Quantum Feature Encoding: To convert data to Quantum States we scale and normalize our data. We then, once again use a ZZ feature map to map our data to the Hilbert Space

3. Kernel Creation: Instead of training a quantum neural network, we have a quantum kernel matrix be computed to measure similarity between data points. This kernel captures nonlinear relationships between unemployment patterns in quantum feature space.

4. Classical Optimization: Finally, we use classical optimization through a SVM using a precomputed kernel. The SVM solves a convex optimization problem to find the optimal decision boundary between classes so we can make our hybrid pipeline.

5. Testing/Conclusions: Finally, we evaluate performance on the held-out test set and review differences in accuracy. We also are able to track the loss function of our quantum algorithm. In turn, we create a final report that lets us know how our models behave and what are main sources of error would have been.