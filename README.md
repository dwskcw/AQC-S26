# AQC-S26

## Dataset Description:
Make_moons is a (by default) 100-sample dataset included as part of Python’s scikit-learn machine learning library. When plotted, the samples produce two interleaving half-circles across a horizontal range of -1.0 to 2.0 and vertical range of -0.5 to 1.0. The library provides options to add Gaussian noise to the samples, which will randomly scatter the samples by a set amount, and additionally we can shuffle the order of the samples.

## Project Hypothesis:
We predict that, under current quantum machine learning capabilities, the classical SVM model will achieve higher accuracy than the quantum classifier on the Two Moons dataset. This is primarily due to severe limitations of quantum hardware and simulation, which hinder the quantum model's ability to detect nonlinear decision boundaries. By directly comparing the two models, we can assess the present limits of quantum machine learning in simple nonlinear tasks, using the Two Moons benchmark.

## QML method:
- To test our hypothesis that classical Support Vector Machines (SVM) models outperform quantum classifiers, we will implement our own Quantum Machine Learning (QML) classifier using a quantum kernel.

1. Preparing Data: Since our dataset will be the Two Moons dataset, our data consists of the coordinates of each point and a binary label of which moon it belongs to (0 = bottom moon, 1 = top moon).  To begin, we will split our data so 70% will be the training set and 30% belong to the testing set, and we will normalize the coordinates.
   
2. Quantum Feature Encoding: Using the ZZ feature map, we will map our normalized classical data points into a quantum state, effectively transforming our 2d points into a quantum Hilbert Space.

3. Kernel Approach: Next, we will compute a quantum kernel matrix to measure the similarity of points in the quantum feature space. Since we have a few features and parameters, training shouldn’t be an issue.

4. Classical Optimization: Once we compute the kernel matrix, we need to send it to a classical SVM. Our SVM uses the kernel matrix to learn an optimal decision boundary between our two moons. This must be done because quantum computing hardware is not yet capable of solving convex optimization models in a timely or reliable manner.

5. Evaluations & Comparison: After the training, we will test our model on our training set. Accuracy will be our metric of success. To compare results, we will also train a classical SVM with an RBF kernel on the same dataset and compare classification accuracy.

6. Final Report: Based on our findings, we will be able to assess our quantum kernel classifier on the moons dataset compared to a classical SVM. Through this process, we will justify or disprove our hypothesis. To conclude our analysis, we will discuss the main sources of error in the quantum model, which will include noise, limited qubit encoding, and possible issues with our feature map.
