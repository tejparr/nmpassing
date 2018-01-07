function [beliefs] = var_algo(obs, n_states, p_trans)
    %variational massage passing
    
    %set prediction length
    if nargin < 4
        t_pred = 10;
    else
        t_pred = varargin{4};
    end
    
    p = linspace(1/(2*n_states), 1-1/(2*n_states), n_states)';
    lst = log( eye(n_states)*(1-p_trans) + ...
        p_trans*(ones(n_states)-eye(n_states))/(n_states-1) );
    
    T = length(obs);
       
    beliefs = ones(n_states, T+t_pred+1)/n_states;
    
    for i = 1:10
        pre = beliefs;
        for t = 1:T+t_pred+1
            %log observation likelihood
            if t == 1
                ll = ones(n_states,1);
            elseif t > T+1
                ll = ones(n_states,1);
            else
                ll = obs(t-1)*log(p) + (1-obs(t-1))*log(1-p);
            end
                
            if t == 1
                forw = log(ones(n_states,1)/n_states);
            else
                forw = lst*pre(:,t-1);
            end
            
            if t == T+t_pred+1
                back = ones(n_states,1);
            else
                back = lst'*pre(:, t+1);
            end

            lb = ll + forw + back;
            beliefs(:, t) =  exp(lb - max(lb));
            beliefs(:, t) = beliefs(:, t)/sum(beliefs(:, t));
        end
    end
    beliefs(:,1) = [];
end
