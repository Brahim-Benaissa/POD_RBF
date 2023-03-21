% https://doi.org/10.1007/s00158-016-1400-y
% https://doi.org/10.1016/j.jocs.2021.101451 

function Norm_Parameters = NormalizeModelParameters(Input_param, Model_Parameters)
    % Get minimum and maximum values for each parameter
    minP = min(Model_Parameters);
    maxP = max(Model_Parameters);
    
    % Normalize model parameters
    Norm_Parameters = (Input_param - minP) ./ (maxP - minP);
end