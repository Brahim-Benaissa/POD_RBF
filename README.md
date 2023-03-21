
The code performs a reduced order modeling using Proper Orthogonal Decomposition (POD) and Radial Basis Function (RBF) interpolation.

The code starts by loading the model parameters and data from two separate text files. It then selects the acceptable reconstruction tolerance value and RBF function along with the RBF parameter for tuning the interpolation to the problem.

Next, the code randomly selects parameters not included in the dataset for which the prediction is needed. It then normalizes the model parameters, performs POD, calculates the POD basis and the amplitudes, performs the truncation, and calculates the reduced equivalents of the POD basis and the amplitudes.

1. Build_PODRBF_Model generates the coefficients matrix and the reduced coefficients matrix, and generates RBF interpolation parameters according to the selected RBF function. It then makes a prediction based on all modes and based on the reduced modes.

Then plots the selected modes, calculates the mean absolute prediction error, mean squared prediction error, and root mean squared prediction error, and then plots the prediction errors in a bar graph. Finally, the code plots the RBF function and saves the reduced model to a text file.


2. USE_Reduced_PODRBF_Model for making predictions using a reduced model with radial basis function (RBF) interpolation. The script loads the parameters and the reduced model from two text files, then selects an RBF function and a gamma value (a parameter for tuning the interpolation to the problem). The script then generates 10 sets of input parameters (randomly selected within the range of the loaded parameters), normalizes them using a NormalizeModelParameters function, and calculates RBF interpolation parameters using an RBF_interpolation function. The Reduced_Model is then used to make predictions based on the RBF interpolation parameters, and the results are displayed in a plot. The script also plots the selected RBF function using an RBF_Fnc_Display function.


 **YUKI Algorithm and POD-RBF for Elastostatic and dynamic crack identification**. *Journal of Computational Science*. 2021. <a href="https://doi.org/10.1016/j.jocs.2021.101451" target="_blank"> https://doi.org/10.1016/j.jocs.2021.101451 </a>.  
 
 **Crack Identification Using Model Reduction based on Proper Orthogonal Decomposition coupled with Radial Basis Functions**. *Structural and Multidisciplinary Optimization*. 2016. <a href="https://doi.org/10.1007/s00158-016-1400-y" target="_blank"> https://doi.org/10.1007/s00158-016-1400-y </a>