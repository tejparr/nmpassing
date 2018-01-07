function [beliefs] = bp_algo(obs, n_states, trans_p)
    % Belief propagation algorithm. For sequential inference problem, 
    % this corresponds to a forward backward algorithm, 
    % Here we have implemented a specific variant called alpha-gamma algorithm.
    
    %set prediction length
    if nargin < 4
        t_pred = 10;
    else
        t_pred = varargin{4};
    end
    
    p = linspace(1/(2*n_states), 1-1/(2*n_states), n_states)';
    
    state_trans = (1-trans_p)*eye(n_states)+...
        trans_p*(ones(n_states) - eye(n_states))/(n_states-1);
    
    T = length(obs);

    alphas = zeros(n_states, T+1);
    pred = zeros(n_states, T+t_pred+1);
    gammas = zeros(n_states, T+1);
    alphas(:,1) = ones(n_states,1)/n_states;
    
    %forward pass
    for t = 2:T+1
        %observation likelihood
        ol = p.^obs(t-1).*(1-p).^(1-obs(t-1));
        
        %prior expectations in time step t
        pred(:,t) = state_trans*alphas(:,t-1);

        alphas(:, t) = ol.*pred(:,t);
        alphas(:,t) = alphas(:,t)./sum(alphas(:,t));
    end
    
    %backward pass
    gammas(:,T+1) = alphas(:,T+1);
    for t = 1:T
        k = T+1-t;
        gammas(:,k) = alphas(:,k).*(state_trans'*(gammas(:, k+1)./pred(:,k+1)));
    
    end
    
    % prediction
    pred(:, T+2) = state_trans*gammas(:,T+1); 
    for tau = T+3:T+t_pred+1
       pred(:,tau) = state_trans*pred(:,tau-1); 
    end
    
    
    beliefs = [gammas(:,2:end), pred(:, T+2:end)];
end