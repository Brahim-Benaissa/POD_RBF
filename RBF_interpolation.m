function RBF_interpolation_parameters = RBF_interpolation (Model_Parameters, Norm_Parameters, Fcn, gamma) 

    % Calculate RBF interpolation parameters according to the chosen RBF interpolation function
switch Fcn
    
    case 'F1'   % Using the Identity RBF function
        RBF_interpolation_parameters = sqrt(sum((Norm_Parameters - NormalizeModelParameters(Model_Parameters, Model_Parameters)).^2, 2))';

    case 'F2'  % Using the Gaussian RBF function
        RBF_interpolation_parameters = exp(-gamma * sum((Norm_Parameters - NormalizeModelParameters(Model_Parameters, Model_Parameters)).^2, 2))';

    case 'F3' % Using the Multiquadric RBF function
        RBF_interpolation_parameters = sqrt(sum((Norm_Parameters - NormalizeModelParameters(Model_Parameters, Model_Parameters)).^2, 2) + gamma^2)';

    case 'F4' % Using the Inverse Multiquadric RBF function
        RBF_interpolation_parameters = 1 ./ sqrt(sum((Norm_Parameters - NormalizeModelParameters(Model_Parameters, Model_Parameters)).^2, 2) + gamma^2)';

    case 'F5' %  Using the Laplacian RBF
        RBF_interpolation_parameters = exp(-gamma * sum(abs(Norm_Parameters - NormalizeModelParameters(Model_Parameters, Model_Parameters)), 2))';

    case 'F6' %  Using The Cauchy RBF
        RBF_interpolation_parameters = 1 ./ (1 + gamma * sum((Norm_Parameters - NormalizeModelParameters(Model_Parameters, Model_Parameters)).^2, 2))';

end