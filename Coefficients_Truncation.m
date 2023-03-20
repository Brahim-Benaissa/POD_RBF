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