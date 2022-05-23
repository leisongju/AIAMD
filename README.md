# AIAMD
This repository is the code repository for the neural network part of the paper "Application of Artificial Neural Network in AMD Method".
# Requirements
 * python >= 3.9
 * torch >= 1.10.0
 * pandas >= 1.3.4
 * numpy >= 1.20.3
 * scikit-learn >= 0.24.2
# Configuration and Running
The training data needs to be obtained by solving the matrix determinant using traditional algorithms, and this part of the code is not public. You can refer to 《Generator Coordinate Treatment of Composite Particle Reaction and Molecule-like Structures》.
There are already two trained model files in the repository. The model file in the single folder is used to predict the diagonal terms of the 8B nuclear Hamiltonian matrix, and the model file in the cross folder is used to predict the off-diagonal terms of the 8B nuclear Hamiltonian.