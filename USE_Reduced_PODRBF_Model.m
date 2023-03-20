clc
clear 

% Recalculate from the saved reduced model (without the original data set)  
%% Select the saved reduced model and the paramteres
Reduced_Prediction=[];

% Load model parameters 
Model_Parameters = load('Parameters.txt');

% Load the reduced
Reduced_Model = load('Reduced_Model.txt');

% Select the RBF function 
% F1: Identity RBF, F2: Gaussian RBF, F3:Multiquadric RBF , F4:Inverse Multiquadric RBF, F5:Laplacian RBF, F6:Cauchy RBF, 
Fcn = strcat('F4');  % (SAME AS THE FUNCTION USED FOR MODEL BUILDING)

% gamma is the RBF paramteres for tuning the interpolation to the problem [0-1]
gamma= 0.5; % (SAME VALUE USED FOR MODEL BUILDING)


% F1: Identity RBF, F2: Gaussian RBF, F3:Multiquadric RBF , F4:Inverse Multiquadric RBF, F5:Laplacian RBF, F6:Cauchy RBF, 
%(tuning paramteres in RBF_interpolation function)

% Select parameters not included in the dataset, forwhich the prediction is needed
% you may connect to an automatic system for other functions or for optimization 
% by setting these values as variables

for i=1:10 % 10 predictions example 

Input_param=min(Model_Parameters) + (max(Model_Parameters)-min(Model_Parameters))*rand;

%% Use the selected model

% Normalize model parameters
Norm_Parameters = NormalizeModelParameters(Input_param,Model_Parameters);

% Generate RBF interpolation parameters according to the selected RBF function
RBF_interpolation_parameters = RBF_interpolation (Model_Parameters, Norm_Parameters, Fcn, gamma);

%% Prediction resuls display 

% Make prediction based on the reduced modes
Reduced_Prediction(:,i) = Reduced_Model*RBF_interpolation_parameters';

plot(Reduced_Prediction,'o-','LineWidth', 2)
xlabel('Index')
ylabel('Value')
title('Prediction from the reduced model')
hold on

pause(0.1)
end

% Plot the RBF FUNCTION
RBF_Fnc_Display (Fcn, gamma) 

