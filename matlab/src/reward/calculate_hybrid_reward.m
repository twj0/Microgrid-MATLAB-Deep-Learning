function [reward, components] = calculate_hybrid_reward(action_W, net_load_W, P_grid_W, price_USD_per_kWh, soc_percent, fuzzy_system)
%CALCULATE_HYBRID_REWARD Combine tracking and economic objectives.
%   reward = CALCULATE_HYBRID_REWARD(action_W, net_load_W, P_grid_W,
%   price_USD_per_kWh, soc_percent) returns the weighted reward for the
%   current timestep. The optional fuzzy_system input allows injecting
%   expert knowledge when available.
%
%   Inputs:
%       action_W          - Storage power command [W], charge > 0.
%       net_load_W        - Net load [W], load minus PV, deficit > 0.
%       P_grid_W          - Grid exchange power [W], import > 0.
%       price_USD_per_kWh - Electricity price [$/kWh].
%       soc_percent       - Battery state of charge [%], forwarded to FIS.
%       fuzzy_system      - Optional fuzzy controller or callback. Use []
%                           to disable.
%
%   Outputs:
%       reward            - Hybrid reward scalar.
%       components        - (Optional) struct with individual terms.

    arguments
        action_W (1,1) double
        net_load_W (1,1) double
        P_grid_W (1,1) double
        price_USD_per_kWh (1,1) double
        soc_percent (1,1) double
        fuzzy_system = []
    end

    % Fixed weights; adjust here if different emphasis is required.
    w_tracking = 1.0;
    w_economic = 0.5;

    trackingReward = calculate_tracking_reward(action_W, net_load_W);
    economicReward = calculate_economic_reward(P_grid_W, price_USD_per_kWh);

    fuzzyCorrection = 0.0;
    if ~isempty(fuzzy_system)
        fuzzyCorrection = evaluate_fuzzy_correction(fuzzy_system, soc_percent, price_USD_per_kWh, net_load_W);
    end

    reward = w_tracking * trackingReward + ...
             w_economic * economicReward + ...
             fuzzyCorrection;

    if nargout > 1
        components = struct( ...
            'tracking', trackingReward, ...
            'economic', economicReward, ...
            'fuzzy', fuzzyCorrection, ...
            'weights', struct('tracking', w_tracking, 'economic', w_economic));
    end
end

function correction = evaluate_fuzzy_correction(fuzzy_system, soc_percent, price_USD_per_kWh, net_load_W)
%EVALUATE_FUZZY_CORRECTION Evaluate fuzzy adjustment without try/catch.
    inputs = [soc_percent, price_USD_per_kWh, net_load_W / 1000.0];

    if isa(fuzzy_system, 'function_handle')
        correction = fuzzy_system(inputs);
    elseif isa(fuzzy_system, 'mamfis') || isa(fuzzy_system, 'sugfis')
        correction = evalfis(fuzzy_system, inputs);
    elseif isobject(fuzzy_system) && ismethod(fuzzy_system, 'evalfis')
        correction = evalfis(fuzzy_system, inputs);
    elseif isstruct(fuzzy_system) && isfield(fuzzy_system, 'eval')
        correction = fuzzy_system.eval(inputs);
    else
        % Unsupported type: return zero contribution.
        correction = 0.0;
    end
end
