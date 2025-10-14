function total_reward = calculate_economic_reward(P_net_load_W, P_batt_W, price_norm, SOC)
% CALCULATE_BENEFIT_REWARD - A reward function focused on positive reinforcement.
%
% This function is designed to reward beneficial actions rather than just
% penalize costly ones. It identifies and quantifies the value created by
% the BESS in three key areas: peak shaving, valley filling, and battery health.
%
% Inputs:
%   P_net_load_W  - Net load the BESS must handle (P_load - P_pv) [W].
%                   >0 means net consumption; <0 means surplus PV.
%   P_batt_W      - Battery power command [W]. >0 for charging, <0 for discharging.
%   price_norm    - Normalized electricity price [0, 1]. 0=lowest, 1=highest.
%   SOC           - Battery state of charge [%].
%
% Output:
%   total_reward  - A scalar reward signal for the RL agent.

% --- 1. Tunable Parameters (Weights and Thresholds) ---
% These values control the agent's priorities.

% -- Weights --
W_PEAK_SHAVING   = 1.5;   % Primary objective: Highest weight for reducing high net load.
W_VALLEY_FILLING = 1.0;   % Secondary objective: Reward for charging at low prices.
W_SOC_COMFORT    = 0.5;   % Background objective: Encourage staying in a healthy SOC range.
W_DEGRADATION    = 0.2;   % Penalty: Discourage overly aggressive (high power) actions.

% -- Thresholds --
PEAK_LOAD_THRESHOLD_W = 5000;  % [W] Net load above which we consider it a "peak" to be shaved.
LOW_PRICE_THRESHOLD   = 0.3;   % [0,1] Price below which we encourage "valley filling" (charging).

% -- SOC Parameters --
SOC_TARGET    = 50.0; % [%] The ideal center point for SOC.
SOC_STD_DEV   = 30.0; % [%] Defines the width of the "comfort zone". Larger value = wider zone.
BATT_RATED_POWER_W = 10000; % [W] For normalizing the degradation penalty.

% --- 2. Calculate Individual Reward Components ---

% Initialize all reward components to zero.
peak_shaving_reward = 0;
valley_filling_reward = 0;

% (A) HIGH-VALUE REWARD: Peak Shaving
% This reward is granted when the net load is high (a peak) and the battery
% is discharging to reduce it. The reward is scaled by how high the price is.
if P_net_load_W > PEAK_LOAD_THRESHOLD_W && P_batt_W < 0
    % -P_batt_W is the positive power being supplied by the battery.
    % price_norm scales the reward: shaving a peak at high price is more valuable.
    peak_shaving_reward = W_PEAK_SHAVING * (-P_batt_W / 1000) * price_norm;
end

% (B) OPPORTUNITY REWARD: Valley Filling (Arbitrage)
% This reward is granted when the electricity price is low and the battery
% is charging. The reward is scaled by how cheap the power is.
if price_norm < LOW_PRICE_THRESHOLD && P_batt_W > 0
    % (1 - price_norm) is a "cheapness factor". If price is 0.1, factor is 0.9.
    valley_filling_reward = W_VALLEY_FILLING * (P_batt_W / 1000) * (1 - price_norm);
end

% (C) HEALTH REWARD: SOC Comfort Zone
% A Gaussian reward that is maximum at SOC_TARGET and smoothly decreases
% as SOC moves away. This encourages the agent to keep the battery ready.
soc_deviation = SOC - SOC_TARGET;
soc_comfort_reward = W_SOC_COMFORT * exp(- (soc_deviation^2) / (2 * SOC_STD_DEV^2));

% (D) HEALTH PENALTY: Battery Degradation
% A small quadratic penalty on battery power to discourage extreme charge/discharge
% rates, promoting longer battery life.
degradation_penalty = -W_DEGRADATION * (P_batt_W / BATT_RATED_POWER_W)^2;


% --- 3. Summation and Final Safety Check ---

% Combine all components into the final reward signal.
total_reward = peak_shaving_reward + ...
               valley_filling_reward + ...
               soc_comfort_reward + ...
               degradation_penalty;

% HARD SAFETY CONSTRAINT: Override all rewards with a large penalty if SOC
% goes outside the absolute physical/safety limits. This is non-negotiable.
if SOC < -10 || SOC > 110
    total_reward = -100;
end

end