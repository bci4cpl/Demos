classdef Infomax < handle
    %INFOMAX Implements an Infomax network
    %   An over-complete neural network with recurrent connections based on
    %   the information maximization approach.
    %   
    %   Written by Aviv Dotan
    %   11.11.2017
    %   
    %   Based on the paper: 
    %   Shriki, O., Sompolinsky, H., & Lee, D. D. (2001). An information 
    %   maximization approach to overcomplete and recurrent 
    %   representations. In Advances in neural information processing 
    %   systems (pp. 612-618).
    %   URL: 
    %   https://papers.nips.cc/paper/1863-an-information-maximization-approach-to-overcomplete-and-recurrent-representations
    %   
    %   To create a new feed-forward network:
    %   net = Infomax(# of inputs, # of outputs)
    %   
    %   Other optional constructor parameters (by order):
    %   - Learning rate
    %   - Learn the feed-forward connections? (logical)
    %   - Learn the recurrent connections? (logical)
    %   Euler's method's parameters (for the recurrent dynamics): 
    %   -	Max # of iterations
    %   -   alpha (=deltat)
    %   -   Tolerance for convergence check
    %   All optional parameters can be changed later through suitable
    %   properties. 
    %   
    %   To train he network:
    %   net.Learn(inputs matrix)
    %   
    %   To get the network's output for given inputs: 
    %   net.Evaluate(inputs matrix)
    %   
    %   To access the networks feed-forward and recurrent connectivity
    %   matrices, use the properties W and K (respectively). 
    %   By default, the feed-forward connections are initialized based on 
    %   Xavier's initialization, and the recurrent connections are
    %   initialized to zeros. The connectivity can also be changed 
    %   manually; e.g. to reset the recurrent connectivity matrix to zeros
    %   use:
    %   net.K = zeros(net.Outputs, net.Outputs)
    %   For more information about Xavier's initialization, see:
    %   Glorot, X., & Bengio, Y. (2010, March). Understanding the 
    %   difficulty of training deep feedforward neural networks. In 
    %   Proceedings of the Thirteenth International Conference on 
    %   Artificial Intelligence and Statistics (pp. 249-256).
    %   URL: http://proceedings.mlr.press/v9/glorot10a.html
    
    properties (GetAccess = public, SetAccess = immutable)
        
        % Number of input neurons
        Inputs;
        
        % Number of output neurons
        Outputs;
        
    end
    
    properties (Access = public)
        
        % Learning rate
        LearningRate = 0.01;
        
        % Number of iterative solve iterations
        niter = 3000;
        
        % Euler method iterative solution parameter
        alpha = 0.2;
        
        % Iterative solve tolerance factor
        tolfun = 1e-8;
        
        % Learn the feed-forward connections?
        LearnFF = true;
        
        % Learn the recurrent connections?
        LearnRec = false;
        
        % Feed-forward connections
        W;
        
        % Recurrent connections
        K;
        
    end
    
    methods
        
        % Constructor
        function obj = Infomax(inputs, outputs, learningrate, ...
                learnFF, learnRec, niter, alpha, tolfun)
            
            obj = obj@handle;
            
            if (nargin < 2)
                error('Infomax must get the network size. ');
            end
            if (inputs > outputs)
                error('Undercomplete networks are not supported. ');
            end
            
            obj.Inputs              = inputs;
            obj.Outputs             = outputs;
            if (nargin > 2)
                obj.LearningRate    = learningrate;
            end
            if (nargin > 3)
                obj.LearnFF         = learnFF;
            end
            if (nargin > 4)
                obj.LearnRec        = learnRec;
            end
            if (nargin > 5)
                obj.niter           = niter;
            end
            if (nargin > 6)
                obj.alpha           = alpha;
            end
            if (nargin > 7)
                obj.tolfun          = tolfun;
            end
            
            % Initialize connections
            obj.W = XavierInitialize([outputs, inputs]);
            obj.K = zeros(outputs, outputs);
            
        end
        
    end
    
    methods (Access = public)
        
        function [g, gp, gpp] = Evaluate(obj, x)
            %EVALUATE Get an output vector s for an input vector x
            
            % If there are no recurrent connections
            if (~any(obj.K))
                [g, gp, gpp] = gfunc(obj.W*x);
            else
                % For each sample, solve ds/dt = -s + g(Wx + Ks) using 
                % Euler's method and find its stable fixed point
                s   = zeros(obj.Outputs, size(x, 2));
                for i = 1:size(x, 2)
                    s(:,i) = obj.gcalc(x(:,i));
                end
                [g, gp, gpp] = gfunc(obj.W*x + obj.K*s);
            end
            
        end
        
        function Learn(obj, x)
            %LEARN Learn from a samples matrix x
            
            % If no learning is to be done
            if (~obj.LearnFF && ~obj.LearnRec)
                return;
            end
            
            n_samples = size(x, 2);
            
            % Initialize
            dW = zeros(size(obj.W));
            dK = zeros(size(obj.K));
            
            % Get the network's output and the derivatives of g
            [s, gp, gpp] = obj.Evaluate(x);
            
            % For each sample
            for nsample = 1:n_samples
                
                x0      = x(:,nsample);
                s0      = s(:,nsample);
                gp0     = gp(:,nsample);
                gpp0    = gpp(:,nsample);
                
                % Calculate the gradient
                Ginv = diag(1 ./ gp0);
                Phi = inv(Ginv - obj.K);
                Chi = Phi * obj.W;
                Gamma = inv(Chi' * Chi) * Chi' * Phi;
                gamma = diag(Chi * Gamma) .* gpp0 ./ (gp0 .^ 3);
                
                % Aggregate connections' changes
                if (obj.LearnFF)
                    dW = dW + (Gamma' + Phi' * gamma * x0');
                end
                if (obj.LearnRec)
                    dK = dK + ((Chi * Gamma)' + Phi' * gamma * s0');
                end
                
            end
            
            % Change connections
            eta = obj.LearningRate / n_samples;
            if (obj.LearnFF)
                obj.W = obj.W + eta.*dW;
            end
            if (obj.LearnRec)
                obj.K = obj.K + eta.*dK;
                obj.K(1:(obj.Outputs + 1):end) = 0;	% No self-coupling
            end
        end
        
        function [cost] = GetCost(obj, x)
            % GETCOST Get the cost function for a given input
            
            cost = 0;
            
            % Get the network's output and the derivatives of g
            [~, gp] = obj.Evaluate(x);
            
            % For each sample
            for nsample = 1:size(x, 2)
                
                gp0 = gp(:,nsample);
                
                % Evaluate Chi
                Ginv = diag(1 ./ gp0);
                Phi = inv(Ginv - obj.K);
                Chi = Phi * obj.W;
                
                % Aggregate log(det(Chi^T*Chi))
                cost = cost + log(det(Chi' * Chi));
                
            end
            
            cost = -0.5 * cost / size(x, 2);
            
        end
        
    end
    
    methods (Access = private)
        
        function s = gcalc(obj, x)
            %GCALC Solve ds/dt = -s + g(Wx + Ks) using Euler's method and 
            %   find its stable fixed point
            
            h = obj.W*x;
            
            % Initialize condition
            s = gfunc(zeros(obj.Outputs, 1));
            
            % Iteratively solve ds/dt = -s + g(Wx + Ks)
            for i = 1:obj.niter
                
                s1 = gfunc(h + obj.K * s);
                s = obj.alpha .* s1 + (1 - obj.alpha) .* s;
                
                % Check for convergence
                if (max(abs(s1 - s)) < obj.tolfun)
                    break; 
                end
                
            end
            
        end
        
    end
    
end

function A = XavierInitialize(size)
%XAVIERINITIALIZE Initialize a connectivity matrix by Xavier's
%   initialization method

A = (1/sqrt(mean(size))).*randn(size);

end

function [g, gp, gpp] = gfunc(x)
%GFUNC The neurons activation function
%   1 / (1 + exp(-x))

g = 1 ./ (1 + exp(-x));
if (nargout > 1)
    gp  = g .* (1 - g);
    gpp = gp .* (1 - 2 * g);
end

end
