function economicReward = calculate_economic_reward(P_grid_W, price_USD_per_kWh)
%CALCULATE_ECONOMIC_REWARD Compute the economic reward component.
%   economicReward = CALCULATE_ECONOMIC_REWARD(P_grid_W, price_USD_per_kWh)
%   returns the reward associated with buying or selling power at the
%   current price. Positive values represent revenue, negative values
%   represent cost. Internal constants determine penalty strength.
%
%   Inputs:
%       P_grid_W          - Grid exchange power [W], >0 buy, <0 sell.
%       price_USD_per_kWh - Real-time energy price [$/kWh].
%
%   Output:
%       economicReward    - Scalar economic reward contribution.

    arguments
        P_grid_W (1,1) double
        price_USD_per_kWh (1,1) double
    end

    % Tunable constants (adjust in code if needed).
    peakPowerThresholdW = 15e3;  % [W] Peak import threshold.
    highPriceThreshold = 0.40;   % [$/kWh] High price definition.
    penaltyFactor = 5e-8;        % Quadratic penalty scale.
    timeStepHours = 1.0;         % Simulation step duration [h].

    % Core transaction term: revenue when exporting, cost when importing.
    P_grid_kW = P_grid_W / 1000.0;
    linear_reward = -(P_grid_kW * price_USD_per_kWh * timeStepHours);

    % Peak power penalty discourages expensive peak-hour imports.
    peak_penalty = 0.0;
    if (P_grid_W > 0) && (P_grid_W > peakPowerThresholdW) && (price_USD_per_kWh > highPriceThreshold)
        peak_penalty = -penaltyFactor * (P_grid_W ^ 2);
    end

    economicReward = linear_reward + peak_penalty;
end
