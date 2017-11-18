%   A toy model for a visual hypercolumn using an overcomplete recurrent
%   Infomax neural network
%
%   Written by Aviv Dotan
%   18.11.2017
%
%   Based on the paper:
%   Shriki O, Yellin D (2016) Optimal Information Representation and
%   Criticality in an Adaptive Sensory Recurrent Neuronal Network. PLoS
%   computational biology 12(2): e1004698.
%   URL:
%   http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004698

clear;
close all;
clc;

%% Generate input samples

n_samples   = 3000; % Number of training samples

% Generate the data
theta   = (2*pi).*rand(1, n_samples);
R       = abs(1 + 0.1.*randn(1, n_samples));
X0      = [R.*cos(theta); R.*sin(theta)];

% Display the input samples
figure('Name', 'Inputs', ...
    'units', 'Normalized', ...
    'Position', [0.05, 0.05 + 0.15*16/9, 0.3, 0.3*16/9], ...
    'NumberTitle', 'off');
scatter(X0(1,:), X0(2,:), 'k.');
xlabel('$$x_1$$', 'Interpreter', 'latex');
set(gca, 'XTick', []);
ylabel('$$x_2$$', 'Interpreter', 'latex');
set(gca, 'YTick', []);
axis equal;

%% For different input contrasts

for r = 0.1:0.2:0.9
    
    %% Set the input contrast
    
    X = r.*X0;
    
    %% Create an overcomplete Infomax network
    
    Net = Infomax(2, 30, 0.001, false, true, 1e7);
    
    % Set the feed-forward interaction
    theta = (2*pi/Net.Outputs).*(1:Net.Outputs);
    Net.W = [cos(theta); sin(theta)]';
    
    %% Train the network
    
    n_train     = 9375 + 6250*r;        % Number of learning steps
    batch_size  = 1;                    % Number of samples per learning step
    plot_freq   = floor(n_train/10);	% Plot frequency
    
    figure('Name', num2str(r, 'r = %-g'), ...
        'units', 'Normalized', ...
        'Position', [0.4, 0.05, 0.55, 0.45*16/9], ...
        'NumberTitle', 'off');
    for t = 1:n_train
        
        %% Learn
        
        % Choose a random batch
        x = X(:, randperm(n_samples, batch_size));
        
        % Train the network
        Net.Learn(x);
        
        %% Progress plot
        
        % Set the plot frequency
        if ~(rem(t, plot_freq) == 0 || t == n_train)
            continue;
        end
        
        % Get the network's cost
        cost = Net.GetCost(X);
        
        clf;
        
        % Precalculations for the plots
        K       = Net.K;
        ind     = floor(Net.Outputs/2);
        theta   = (360/Net.Outputs).*(1:Net.Outputs) - 180;
        
        % Display title
        subplot(12, 2, 1:2);
        titletext = text(0.5, 0.5, ...
            ['$$r=' num2str(r, '%-g') '$$ , ' ...
            '$$t=' num2str(t, '%-d') '$$ , ' ...
            '$$\eta=' num2str(Net.LearningRate, '%-g') '$$ , ' ...
            '$$\varepsilon=' num2str(cost, '%-g') '$$'], ...
            'HorizontalAlignment',  'center', ...
            'FontSize',             14, ...
            'Interpreter',          'latex');
        axis off;
        
        % Display the recurrent connectivity
        subplot(12, 2, 3:2:11);
        imagesc(Net.K.*Net.Outputs, [-8, 8]);
        colormap('gray');
        caxis([-9, 9]);
        colorbar('Ticks', [-8, 0, 8]);
        set(gca, 'XTick', [1, ind, Net.Outputs], ...
            'XTickLabels', cellstr(num2str([-180; 0; 180], '%-d')));
        xlabel('Pre-synaptic PO [deg]');
        set(gca, 'YTick', [1, ind, Net.Outputs], ...
            'YTickLabels', cellstr(num2str([-180; 0; 180], '%-d')));
        ylabel('Post-synaptic PO [deg]');
        axis xy square;
        
        % Display the recurrent connectivity profile
        subplot(12, 2, 15:2:23);
        profile = Net.K(ind, :);
        profile(ind) = NaN;
        plot(theta, profile.*Net.Outputs, 'Linewidth', 2);
        set(gca, 'XTick', [-180, 0, 180]);
        xlim([-180, 180]);
        xlabel('\Delta PO [deg]');
        ylim([-9, 9]);
        set(gca, 'YTick', [-8, 0, 8]);
        ylabel('Interaction strength');
        
        
        % Display the network's response w/ and w/o recurrent interactions
        subplot(12, 2, 4:2:12);
        stim = [-r; 0];
        resp_w  = Net.Evaluate(stim);
        Net.K = zeros(Net.Outputs);
        resp_wo = Net.Evaluate(stim);
        Net.K = K;
        plot(theta, resp_w, theta, resp_wo, '--', 'Linewidth', 2);
        set(gca, 'XTick', [-180, 0, 180]);
        xlim([-180, 180]);
        xlabel('PO [deg]');
        ylim([0, 1]);
        set(gca, 'YTick', 0:0.2:1);
        ylabel('Response');
        legend({'with int.', 'w/o int.'});
        
        % Display the network's response for different contrast levels
        subplot(12, 2, 16:2:24);
        rs = 0.1:0.2:0.9;
        stim = [-rs; zeros(size(rs))];
        resp = Net.Evaluate(stim);
        cmap = parula(length(rs));
        set(gca, 'ColorOrder', cmap(end:-1:1, :), ...
            'NextPlot', 'replacechildren');
        plot(theta, resp, 'Linewidth', 2);
        set(gca, 'XTick', [-180, 0, 180]);
        xlim([-180, 180]);
        xlabel('PO [deg]');
        ylim([0, 1]);
        set(gca, 'YTick', 0:0.2:1);
        ylabel('Response');
        legend(cellstr(num2str(rs', '$$r=%-g$$')), 'Interpreter', 'latex');
        
        % Draw the plot
        drawnow;
        
    end
    
end
