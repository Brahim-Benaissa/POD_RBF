% https://doi.org/10.1007/s00158-016-1400-y
% https://doi.org/10.1016/j.jocs.2021.101451 

function [POD_basis, Reduced_POD_basis, Amplitudes, Reduced_Amplitudes, Modes, Truncation_index] = POD_Truncation_Amplitudes (Model_Data,Reconstruction_tolerance)
    % Perform POD 
    Covariance = Model_Data * Model_Data';

    % Calculate the eigenvalues
    [V, D] = eig(Covariance);
    Modes=flip(diag(D));

    % Calculate reconstruction error
    cumulative_sum = cumsum(Modes);
    Reconstruction_error = cumulative_sum / sum(Modes); 

    % Get the index for truncation according to the selected error value 
    Truncation_index =find(Reconstruction_error > 1-Reconstruction_tolerance, 1);

    % calculate the phi and A matrices
    POD_basis = fliplr(V);
    Amplitudes = POD_basis' * Model_Data;

    % calculate the reduced phi and A matrices 
    Reduced_POD_basis = POD_basis(:,1:Truncation_index);
    Reduced_Amplitudes = Amplitudes(1:Truncation_index,:);

end