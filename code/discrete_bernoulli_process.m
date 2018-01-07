function [obs, s] = discrete_bernoulli_process(T, n_states, trans_p)
    % Generative process of a bernoulli process with discrete number 
    % of states. 
    % T -> number of time steps
    % n_state -> number of possible distinct probabilities
    % trans_p -> transition probability
    
    %probability of observing o = 1 associated with each state
    p = linspace(1/(2*n_states), 1-1/(2*n_states), n_states);
    
    %observations
    obs = zeros(1, T);
    
    %states
    s = zeros(1,T);
    s(1) = randi(n_states);
    
    for t = 2:T
        if rand() < p(s(t-1))
            obs(t-1) = 1;
        end
        
        if rand() < trans_p
            s(t) = randi(n_states);
        else
            s(t) = s(t-1);
        end
    end
end