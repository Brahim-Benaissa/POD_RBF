% https://doi.org/10.1007/s00158-016-1400-y
% https://doi.org/10.1016/j.jocs.2021.101451 

function [Coefficients, Reduced_Coefficients] = Coefficients_Truncation(Amplitudes, Reduced_Amplitudes, Model_Param)

    % Normalize model parameters
    pNorm = NormalizeModelParameters(Model_Param, Model_Param);

    % Generate distance matrix
    distMat = pdist2(pNorm, pNorm);

    % Generate B matrix
    Coefficients = Amplitudes / distMat;

    % calculate the reduced B matrix
    Reduced_Coefficients = Reduced_Amplitudes / distMat;
end