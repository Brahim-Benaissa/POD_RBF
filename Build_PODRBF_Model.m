clc
clear 

%% select data set, reconstruction tolerance and testing case 

% Load model parameters and equivalent data
Model_Parameters = load('Parameters.txt');
Model_Data = load('Data.txt');

% Select the acceptable reconstruction tolerance value
Reconstruction_tolerance= 1e-6;

% Select the RBF function 
% F1: Identity RBF, F2: Gaussian RBF, F3:Multiquadric RBF , F4:Inverse Multiquadric RBF, F5:Laplacian RBF, F6:Cauchy RBF, 
Fcn = strcat('F4');  

% gamma is the RBF paramteres for tuning the interpolation to the problem [0-1]
gamma= 0.5; 

% randomly Select parameters not included in the dataset, forwhich the prediction is needed
% Within the parameters boundaries you may select different test points and
% different reconstruction tolerance values for building the model

Input_param=min(Model_Parameters) + (max(Model_Parameters)-min(Model_Parameters))*rand;

%% Model Building

% Normalize model parameters
Norm_Parameters = NormalizeModelParameters(Input_param,Model_Parameters);

% Perform POD and calculate the  POD basis and the  Amplitudes, preform the Truncation and calculate the reduced equivalents POD basis and the  Amplitudes
[POD_basis, Reduced_POD_basis, Amplitudes, Reduced_Amplitudes, Modes, Truncation_index] = POD_Truncation_Amplitudes(Model_Data,Reconstruction_tolerance);

% Generate the Coefficients matrix and the reduced Coefficients matrix
[Coefficients, Reduced_Coefficients] = Coefficients_Truncation(Amplitudes, Reduced_Amplitudes, Model_Parameters);

% Generate RBF interpolation parameters according to the selected RBF function
RBF_interpolation_parameters = RBF_interpolation (Model_Parameters, Norm_Parameters, Fcn, gamma);

%% Modes and prediction resuls display 

% Make prediction based on all modes
Prediction = POD_basis*Coefficients*RBF_interpolation_parameters';

% Make prediction based on the reduced modes
Reduced_Prediction = Reduced_POD_basis*Reduced_Coefficients*RBF_interpolation_parameters';

% Plot the selected modes
figure(1)
semilogy(Modes,'o-','LineWidth', 2, 'Color', [0.9290 0.6940 0.1250])
hold on 
semilogy(Modes(1:Truncation_index,1),'o-','LineWidth', 2, 'Color', 'red')
xlabel('Mode Index')
ylabel('Value')
title('Prediction Error: full vs reduced')
xline(Truncation_index, '--k', 'LineWidth', 1.5);
legend('All Modes','Selected Modes','Truncation Point')
text(Truncation_index+2, max(Modes(:,1))/2, num2str(Truncation_index), 'FontSize', 12, 'Color', 'k', 'FontWeight', 'bold');


% Calculate mean absolute Prediction error
Prediction_MAE = mean(abs(Prediction - Reduced_Prediction));

% Calculate mean squared Prediction error
Prediction_MSE = mean((Prediction - Reduced_Prediction).^2);

% Calculate root mean squared Prediction error
Prediction_RMSE = sqrt(Prediction_MSE);

% Plot the Prediction errors in a bar graph
figure(2)
bar([Prediction_MAE, Prediction_MSE, Prediction_RMSE])
xticklabels({'Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'})
ylabel('Error')
title('Prediction Error: full vs reduced')


% Plot the RBF FUNCTION
RBF_Fnc_Display (Fcn, gamma) 

%% Save the reduced model and proceed without the data set

Reduced_Model = Reduced_POD_basis*Reduced_Coefficients;

% Write the Reduced_Model matrix to a text file
dlmwrite('Reduced_Model.txt', Reduced_Model, 'delimiter', '\t');

