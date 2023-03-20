function RBF_Fnc_Display (Fcn, gamma) 

% Define the range of the x and y axes
x = -5:0.01:5;
y = -5:0.01:5;

% Create a grid of points using the meshgrid function
[X,Y] = meshgrid(x,y);

% Define the center and standard deviation of the Gaussian RBF
center = [0, 0];

switch Fcn
    
    case 'F1'   % Using the Identity RBF function
        rbf = sqrt((X-center(1)).^2 + (Y-center(2)).^2);
   
    case 'F2'  % Using the Gaussian RBF function
        rbf = exp(-((X-center(1)).^2 + (Y-center(2)).^2)/(2*gamma^2));

    case 'F3' % Using the Multiquadric RBF function
        rbf = sqrt((X-center(1)).^2 + (Y-center(2)).^2 + gamma^2);

    case 'F4' % Using the Inverse Multiquadric RBF function
        rbf = 1 ./ sqrt((X-center(1)).^2 + (Y-center(2)).^2 + gamma^2);

    case 'F5' %  Using the Laplacian RBF
        gamma = 0.5;
        rbf = abs(X - center(1)) + abs(Y - center(2)) + gamma;
        rbf = gamma./(rbf);

    case 'F6' %  Using The Cauchy RBF
        gamma = 0.5;
        rbf = 1./(1 + ((X-center(1)).^2 + (Y-center(2)).^2)/gamma^2);

end

% Display the 2D shape of the RBF  
figure;
s=surf(X,Y,rbf,'FaceAlpha',1);
s.EdgeColor = 'none';
xlabel('X');
ylabel('Y');
zlabel('RBF');
title('2D Shape of the Selected RBF');