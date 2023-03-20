function Norm_Parameters = NormalizeModelParameters(Input_param, Model_Parameters)
    % Get minimum and maximum values for each parameter
    minP = min(Model_Parameters);
    maxP = max(Model_Parameters);
    
    % Normalize model parameters
    Norm_Parameters = (Input_param - minP) ./ (maxP - minP);
end