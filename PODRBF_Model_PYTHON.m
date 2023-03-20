import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist


def RBF_interpolation(Model_Parameters, Norm_Parameters, Fcn):
    
    def normalize_model_parameters(input_param, model_parameters):
        # Get minimum and maximum values for each parameter
        min_p = np.min(model_parameters)
        max_p = np.max(model_parameters)
    
        # Normalize model parameters
        norm_parameters = (input_param - min_p) / (max_p - min_p)
    
        return norm_parameters
    
    # Calculate RBF interpolation parameters according to the chosen RBF interpolation function
    switcher = {
        'F1': lambda x: np.sqrt(np.sum((Norm_Parameters - NormalizeModelParameters(Model_Parameters, Model_Parameters))**2, axis=1)).T,
        'F2': lambda x: np.exp(-0.3 * np.sum((Norm_Parameters - NormalizeModelParameters(Model_Parameters, Model_Parameters))**2, axis=1)).T,
        'F3': lambda x: np.sqrt(np.sum((Norm_Parameters - NormalizeModelParameters(Model_Parameters, Model_Parameters))**2, axis=1) + 0.5**2).T,
        'F4': lambda x: 1 / np.sqrt(np.sum((Norm_Parameters - NormalizeModelParameters(Model_Parameters, Model_Parameters))**2, axis=1) + 1**2).T,
        'F5': lambda x: np.exp(-0.15 * np.sum(np.abs(Norm_Parameters - NormalizeModelParameters(Model_Parameters, Model_Parameters)), axis=1)).T,
        'F6': lambda x: 1 / (1 + 0.5 * np.sum((Norm_Parameters - NormalizeModelParameters(Model_Parameters, Model_Parameters))**2, axis=1)).T,
    }
    
    # Get the function based on the input Fcn
    func = switcher.get(Fcn)
    
    # Calculate the RBF interpolation parameters
   
  

def POD_Truncation_Amplitudes(Model_Data, Reconstruction_tolerance):
    # Perform POD 
    Covariance = np.dot(Model_Data, Model_Data.T)

    # Calculate the eigenvalues
    D, V = np.linalg.eig(Covariance)
    Modes = np.flip(np.sort(np.real(D)))

    # Calculate reconstruction error
    cumulative_sum = np.cumsum(Modes)
    Reconstruction_error = cumulative_sum / np.sum(Modes)

    # Get the index for truncation according to the selected error value 
    Truncation_index = np.where(Reconstruction_error > 1-Reconstruction_tolerance)[0][0]

    # calculate the phi and A matrices
    POD_basis = np.fliplr(V)
    Amplitudes = np.dot(POD_basis.T, Model_Data)

    # calculate the reduced phi and A matrices 
    Reduced_POD_basis = POD_basis[:,0:Truncation_index+1]
    Reduced_Amplitudes = Amplitudes[0:Truncation_index+1,:]

    return POD_basis, Reduced_POD_basis, Amplitudes, Reduced_Amplitudes, Modes, Truncation_index



def normalize_model_parameters(input_param, model_parameters):
    # Get minimum and maximum values for each parameter
    min_p = np.min(model_parameters)
    max_p = np.max(model_parameters)

    # Normalize model parameters
    norm_parameters = (input_param - min_p) / (max_p - min_p)

    return norm_parameters


   
def Coefficients_Truncation(Amplitudes, Reduced_Amplitudes, Model_Param):

    # Normalize model parameters
    pNorm = NormalizeModelParameters(Model_Param, Model_Param)

    # Generate distance matrix
    distMat = pdist(pNorm)

    # Reshape the distance matrix into a square matrix
    distMat = np.reshape(distMat, (len(Model_Param), len(Model_Param)))

    # Generate B matrix
    Coefficients = Amplitudes / distMat

    # Calculate the reduced B matrix
    Reduced_Coefficients = Reduced_Amplitudes / distMat

    return Coefficients, Reduced_Coefficients



# select data set, reconstruction tolerance and testing case
# Load model parameters and equivalent data
Model_Parameters = np.loadtxt('Parameters.txt')
Model_Data = np.loadtxt('Data.txt')

# Select the acceptable reconstruction tolerance value
Reconstruction_tolerance = 1e-6

# Select the RBF function
Fcn = 'F1'
# F1: Identity RBF, F2: Gaussian RBF, F3:Multiquadric RBF , F4:Inverse Multiquadric RBF, F5:Laplacian RBF, F6:Cauchy RBF,
# (tuning paramteres in RBF_interpolation function)

# randomly Select parameters not included in the dataset, forwhich the prediction is needed
# Within the parameters boundaries you may select different test points and
# different reconstruction tolerance values for building the model

Input_param = np.random.uniform(low=np.min(Model_Parameters), high=np.max(Model_Parameters))

# Model Building

# Normalize model parameters
Norm_Parameters = NormalizeModelParameters(Input_param, Model_Parameters)

# Perform POD and calculate the  POD basis and the  Amplitudes, preform the Truncation and calculate the reduced equivalents POD basis and the  Amplitudes
POD_basis, Reduced_POD_basis, Amplitudes, Reduced_Amplitudes, Modes, Truncation_index = POD_Truncation_Amplitudes(Model_Data, Reconstruction_tolerance)

# Generate the Coefficients matrix and the reduced Coefficients matrix
Coefficients_Truncation(Amplitudes, Reduced_Amplitudes, Model_Param)

# Generate RBF interpolation parameters according to the selected RBF function
RBF_interpolation(Model_Parameters, Norm_Parameters, Fcn)


# Modes and prediction results display

# Make prediction based on all modes
Prediction = POD_basis @ Coefficients @ RBF_interpolation_parameters.T

# Make prediction based on the reduced modes
Reduced_Prediction = Reduced_POD_basis @ Reduced_Coefficients @ RBF_interpolation_parameters.T

# Plot the selected modes
plt.figure(1)
plt.semilogy(Modes,'o-', linewidth=2, color=[0.9290, 0.6940, 0.1250])
plt.hold(True)
plt.semilogy(Modes[0:Truncation_index,0],'o-', linewidth=2, color='red')
plt.xlabel('Mode Index')
plt.ylabel('Value')
plt.title('Prediction Error: full vs reduced')
plt.axvline(x=Truncation_index, linestyle='--', color='k', linewidth=1.5)
plt.legend(['All Modes', 'Selected Modes', 'Truncation Point'])
plt.text(Truncation_index+2, max(Modes[:,0])/2, str(Truncation_index), fontsize=12, color='k', fontweight='bold')

# Calculate mean absolute Prediction error
Prediction_MAE = np.mean(np.abs(Prediction - Reduced_Prediction))

# Calculate mean squared Prediction error
Prediction_MSE = np.mean((Prediction - Reduced_Prediction)**2)

# Calculate root mean squared Prediction error
Prediction_RMSE = np.sqrt(Prediction_MSE)

# Plot the Prediction errors in a bar graph
plt.figure(2)
plt.bar(['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'], [Prediction_MAE, Prediction_MSE, Prediction_RMSE])
plt.ylabel('Error')
plt.title('Prediction Error: full vs reduced')

# Save the reduced model and proceed without the data set
Reduced_Model = Reduced_POD_basis @ Reduced_Coefficients

# Write the Reduced_Model matrix to a text file
np.savetxt('Reduced_Model.txt', Reduced_Model, delimiter='\t')









# Recalculate from the saved reduced model (without the original data set)
## Select the saved reduced model and the parameters
Reduced_Prediction = []

# Load model parameters
Model_Parameters = np.loadtxt('Parameters.txt')

# Load the reduced model
Reduced_Model = np.loadtxt('Reduced_Model.txt')

# Select the RBF function (THE FUNCTION USED FOR MODEL BUILDING)
Fcn = 'F1'

# F1: Identity RBF, F2: Gaussian RBF, F3:Multiquadric RBF , F4:Inverse Multiquadric RBF, F5:Laplacian RBF, F6:Cauchy RBF, 
# (tuning parameters in RBF_interpolation function)

# Select parameters not included in the dataset, for which the prediction is needed
# you may connect to an automatic system for other functions or for optimization 
# by setting these values as variables

for i in range(10): # 10 predictions example 

    Input_param = np.random.uniform(Model_Parameters.min(), Model_Parameters.max())

    ## Use the selected model

    # Normalize model parameters
    Norm_Parameters = (Input_param - Model_Parameters.min()) / (Model_Parameters.max() - Model_Parameters.min())

    # Generate RBF interpolation parameters according to the selected RBF function
    RBF_interpolation_parameters = RBF_interpolation(Model_Parameters, Norm_Parameters, Fcn)

    ## Prediction results display

    # Make prediction based on the reduced model
    Reduced_Prediction.append(Reduced_Model @ RBF_interpolation_parameters)

    plt.plot(Reduced_Prediction[-1],'o-',linewidth=2)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Prediction from the reduced model')
    plt.pause(0.1)

plt.show()