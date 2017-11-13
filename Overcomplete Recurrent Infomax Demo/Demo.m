%   A simple geometrical demonstration of an overcomplete Infomax network
%   
%   Written by Aviv Dotan
%   11.11.2017
%   
%   Based on the paper: 
%   Shriki, O., Sompolinsky, H., & Lee, D. D. (2001). An information 
%   maximization approach to overcomplete and recurrent representations. In
%   Advances in neural information processing systems (pp. 612-618).
%   URL: 
%   https://papers.nips.cc/paper/1863-an-information-maximization-approach-to-overcomplete-and-recurrent-representations

clear;
close;
clc;

%% Uniformly sample 2D points from a hexagon

n_samples = 3000; % Number of training points

% Generate the data
X = rand(2, n_samples);

% Reshape the data distribution into a hexagon
n3              = floor(n_samples/3);
ind3            = 1:n3;
R               = @(theta) [ cos(theta), sin(theta);
                            -sin(theta), cos(theta)];
D               = diag([sqrt(3)/sqrt(2), sqrt(2)/2]);
mu              = [-sqrt(3)/2; 1/2];
X               = D*R(pi/4)*X + mu;             % Reshape into a Rhombus
X(:, ind3)      = R(2*pi/3)*X(:, ind3);         % Rotate third of the data
X(:, ind3 + n3) = R(4*pi/3)*X(:, ind3 + n3);	% Rotate third of the data

%% Create an overcomplete Infomax network

Net = Infomax(2, 3);

%% Train the network

n_train     = n_samples;            % Number of learning steps
batch_size  = 1;                    % Number of samples per learning step
plot_freq   = floor(n_train/100);   % Plot frequency

figure();
for t = 1:n_train
    
    % Choose a random batch
    x = X(:, randperm(n_samples, batch_size));
    
    % Train the network
    Net.Learn(x);
    
    % Set the plot frequency
    if ~(rem(t, plot_freq) == 0 || t == n_train)
        continue;
    end
    
    % Get the network's axes (normalized)
    Wpinv = pinv(Net.W);
    Wpinv = Wpinv ./ sqrt((4/3)*max(diag(Wpinv'*Wpinv)));
    
    % Plot the training data
    scatter(X(1,:), X(2,:), 'k.');
    hold on;
    
    % Plot the network's axes
    quiver(zeros(1,3), zeros(1,3), Wpinv(1,:), Wpinv(2,:), ...
        'b', 'Linewidth', 2);
    hold off;
    
    % Plot formatting
    title(sprintf('$$t=%-d$$', t), 'Interpreter', 'latex');
    xlim([-1, 1]);
    xticks([]);
    xlabel('$$x_1$$', 'Interpreter', 'latex');
    ylim([-1, 1]);
    yticks([]);
    ylabel('$$x_2$$', 'Interpreter', 'latex');
    axis square;
    
    % Draw the plot
    drawnow;
    pause(0.01);
    
end
