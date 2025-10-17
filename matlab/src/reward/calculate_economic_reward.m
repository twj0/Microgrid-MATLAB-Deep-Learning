function total_reward = calculate_economic_reward(P_net_load_W, P_batt_W, price_norm, SOC)
% CALCULATE_ECONOMIC_REWARD - A reward function for economic optimization.
%
% This function calculates a reward for the RL agent based on economic criteria.
% It considers grid interaction, electricity prices, and battery health.
%
% Inputs:
%   P_net_load_W  - Net load the BESS must handle (P_load - P_pv) [W].
%                   >0 means net consumption; <0 means surplus PV.
%   P_batt_W      - Battery power command [W]. >0 for charging, <0 for discharging.
%   price_norm    - Normalized electricity price. Either 0 (low) or 1 (high).
%   SOC           - Battery state of charge [%].
%
% Output:
%   total_reward  - A scalar reward signal for the RL agent.

% --- 1. Tunable Parameters (Weights and Thresholds) ---
% These values control the agent's priorities.

% -- Weights --
W_PEAK_SHAVING   = 2.0;   % Reward for reducing demand during high-price periods.
W_VALLEY_FILLING = 1.5;   % Reward for charging at low prices.
W_GRID_SELLING   = 1.0;   % Reward for selling energy back to the grid.
W_SOC_COMFORT    = 0.5;   % Encourage staying in a healthy SOC range.
W_DEGRADATION    = 0.3;   % Penalty for aggressive charge/discharge actions.

% -- Thresholds --
PEAK_LOAD_THRESHOLD_W = 4000;  % [W] Net load threshold for peak shaving.
HIGH_PRICE_VALUE      = 1;     % Value indicating high price.
LOW_PRICE_VALUE       = 0;     % Value indicating low price.

% -- SOC Parameters --
SOC_TARGET      = 50.0;  % [%] The ideal center point for SOC.
SOC_STD_DEV     = 20.0;  % [%] Width of the comfort zone.
SOC_UPPER_LIMIT = 100;   % [%] Upper limit for SOC.
SOC_LOWER_LIMIT = 0;     % [%] Lower limit for SOC.
BATT_RATED_POWER_W = 10000; % [W] For normalizing the degradation penalty.

% --- 2. Calculate Grid Power ---
P_grid_W = P_net_load_W + P_batt_W;

% --- 3. Calculate Individual Reward Components ---

% Initialize all reward components to zero.
peak_shaving_reward = 0;
valley_filling_reward = 0;
grid_selling_reward = 0;

% (A) HIGH-VALUE REWARD: Peak Shaving
% Reward for discharging during high-price periods to reduce grid consumption.
if P_net_load_W > PEAK_LOAD_THRESHOLD_W && P_batt_W < 0 && price_norm == HIGH_PRICE_VALUE
    % The higher the discharge power, the greater the reward (but still negative)
    peak_shaving_reward = W_PEAK_SHAVING * (-P_batt_W / 1000);
end

% (B) OPPORTUNITY REWARD: Valley Filling (Arbitrage)
% Reward for charging at low prices to store energy.
if price_norm == LOW_PRICE_VALUE && P_batt_W > 0
    valley_filling_reward = W_VALLEY_FILLING * (P_batt_W / 1000);
end

% (C) GRID SELLING REWARD
% Reward for selling energy back to the grid (when P_grid < 0).
if P_grid_W < -100  % Small tolerance to avoid rewarding noise
    grid_selling_reward = W_GRID_SELLING * (-P_grid_W / 1000);
end

% (D) HEALTH REWARD: SOC Comfort Zone
% A Gaussian reward that is maximum at SOC_TARGET and smoothly decreases
% as SOC moves away. This encourages the agent to keep the battery ready.
if SOC >= SOC_LOWER_LIMIT && SOC <= SOC_UPPER_LIMIT
    soc_deviation = SOC - SOC_TARGET;
    soc_comfort_reward = W_SOC_COMFORT * exp(- (soc_deviation^2) / (2 * SOC_STD_DEV^2));
else
    % Strong penalty for violating SOC limits
    soc_comfort_reward = -50;
end

% (E) HEALTH PENALTY: Battery Degradation
% A small quadratic penalty on battery power to discourage extreme charge/discharge
% rates, promoting longer battery life.
degradation_penalty = -W_DEGRADATION * (abs(P_batt_W) / BATT_RATED_POWER_W)^2;

% --- 4. Summation and Final Safety Check ---

% Combine all components into the final reward signal.
total_reward = peak_shaving_reward + ...
               valley_filling_reward + ...
               grid_selling_reward + ...
               soc_comfort_reward + ...
               degradation_penalty;

% HARD SAFETY CONSTRAINT: Override all rewards with a large penalty if SOC
% goes outside operational limits.
if SOC < SOC_LOWER_LIMIT || SOC > SOC_UPPER_LIMIT
    total_reward = -100;
end

end