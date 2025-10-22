function penalty = Battery_Health_Penalty(P_batt_norm, SOC, dSOC_dt, dSOH_dt)
%BATTERY_HEALTH_PENALTY Battery health shaping with segmented penalties/rewards.
%   penalty = Battery_Health_Penalty(P_batt_norm, SOC, dSOC_dt, dSOH_dt)
%
%   Complex combination of elementary functions that encourages the agent
%   to keep the battery within healthy operating envelopes while exploring
%   nuanced behaviours. The output mixes penalties (negative) and small
%   rewards (positive) but is primarily punitive whenever constraints are
%   violated.
%
%   Design goals (per user specification, 2025-10-19):
%     • SOC ∈ [0, 100] receives a Gaussian-shaped reward centred at 50
%       with magnitude ≈ 1/5 of the economic reward (~32).
%     • SOC outside [0, 100] incurs penalties more severe than any
%       achievable economic reward (> 160 in magnitude).
%     • SOH derivative (|dSOH_dt| ≤ 1e-5) provides gentle rewards when the
%       trend is favourable (negative drift) and penalties otherwise.
%     • dSOC_dt and battery power alignments add dynamic penalties.
%     • Output is bounded to [-320, 32] to keep values numerically stable.

%% ------------------------------------------------------------------------
% 1. Input validation and normalisation
%% ------------------------------------------------------------------------

if any(~isfinite([P_batt_norm, SOC, dSOC_dt, dSOH_dt]))
    penalty = 0.0;
    return;
end

% Avoid numerical overflow yet allow very large excursions for SOC.
SOC_eval   = bound_value(SOC,  -800.0,  1200.0);
dSOC_eval  = bound_value(dSOC_dt, -500.0, 500.0);
dSOH_eval  = bound_value(dSOH_dt, -5e-5, 5e-5);
abs_power  = abs(P_batt_norm);

%% ------------------------------------------------------------------------
% 2. Component calculations
%% ------------------------------------------------------------------------

soc_component      = compute_soc_component(SOC_eval);
power_component    = compute_power_component(P_batt_norm, abs_power);
dynamics_component = compute_dynamics_component(P_batt_norm, dSOC_eval);
soh_component      = compute_soh_component(dSOH_eval, SOC_eval);
synergy_component  = compute_synergy(abs_power, SOC_eval, dSOC_eval);
curiosity_component = compute_curiosity(P_batt_norm, SOC_eval, dSOC_eval);

%% ------------------------------------------------------------------------
% 3. Aggregate and clamp
%% ------------------------------------------------------------------------

raw_penalty = soc_component + power_component + dynamics_component + ...
              soh_component + synergy_component + curiosity_component;

penalty = clamp_reward(raw_penalty, 320.0, 32.0);

end

%% ------------------------------------------------------------------------
% Component helper functions
%% ------------------------------------------------------------------------

function value = compute_soc_component(SOC)
    GAUSS_PEAK   = 64.0;
    GAUSS_SIGMA  = 12.0;
    EDGE_QUAD_GAIN = 0.045;

    OUT_BASE     = 185.0;
    OUT_POWER_GAIN = 0.12;
    OUT_CLAMP      = -320.0;

    if SOC >= 0.0 && SOC <= 100.0
        centered = SOC - 50.0;
        gaussian_reward = GAUSS_PEAK * exp(-(centered / GAUSS_SIGMA) ^ 2);
        edge_penalty = -EDGE_QUAD_GAIN * (centered ^ 2);
        value = bounded_value(gaussian_reward + edge_penalty, -36.0, GAUSS_PEAK);
        return;
    end

    if SOC < 0.0
        deficit = -SOC;
        penalty = -OUT_BASE - OUT_POWER_GAIN * (deficit ^ 2);
    else
        excess = SOC - 100.0;
        penalty = -OUT_BASE - OUT_POWER_GAIN * (excess ^ 2);
    end
    value = max(penalty, OUT_CLAMP);
end

function value = compute_power_component(P_batt_norm, abs_power)
    SAFE_GAIN     = 6.5;
    SAFE_SIGMA    = 0.30;
    MODERATE_GAIN = -18.0;
    HIGH_GAIN     = -42.0;
    EXTREME_GAIN  = -55.0;

    safe_bonus = SAFE_GAIN * exp(-(abs_power / SAFE_SIGMA) ^ 2);
    moderate_penalty = MODERATE_GAIN * (smoothstep(abs_power, 0.35, 0.85) ^ 1.1);
    high_penalty = HIGH_GAIN * (smoothstep(abs_power, 0.75, 1.25) ^ 1.4);
    extreme_penalty = EXTREME_GAIN * log1p(max(abs_power - 1.40, 0.0) * 3.2);
    resonance = 5.0 * sin(1.8 * P_batt_norm) * exp(-(abs_power / 1.6) ^ 2);

    combined = safe_bonus + moderate_penalty + high_penalty + extreme_penalty + resonance;
    value = bounded_value(combined, -120.0, 18.0);
end

function value = compute_dynamics_component(P_batt_norm, dSOC_dt)
    rate = abs(dSOC_dt);
    calm_bonus = 8.0 * exp(-(rate / 6.0) ^ 2);
    rate_penalty = -24.0 * ((rate / 9.0) ^ 1.12) / (1.0 + (rate / 32.0) ^ 1.3);
    alignment = -20.0 * tanh((P_batt_norm * dSOC_dt) / 120.0);
    oscillation = 4.0 * sin(0.18 * dSOC_dt) * exp(-(rate / 48.0) ^ 2);

    value = bounded_value(calm_bonus + rate_penalty + alignment + oscillation, -75.0, 14.0);
end

function value = compute_soh_component(dSOH_dt, SOC)
    HEALTH_CENTER = -2.5e-6;
    HEALTH_SIGMA  = 1.2e-6;
    HEALTH_GAIN   = 24.0;
    PENALTY_GAIN  = 220000.0;  % converts >0 drift into strong penalty

    healthy_reward = HEALTH_GAIN * exp(-((dSOH_dt - HEALTH_CENTER) / HEALTH_SIGMA) ^ 2);

    if dSOH_dt > 0
        drift_penalty = -PENALTY_GAIN * (dSOH_dt ^ 2);
    else
        drift_penalty = 0.0;
    end

    soc_focus = smoothstep(abs(SOC - 50.0), 25.0, 90.0);
    shaping = (1.0 - soc_focus) * 6.0;

    value = bounded_value(healthy_reward + drift_penalty + shaping, -45.0, 24.0);
end

function value = compute_synergy(abs_power, SOC, dSOC_dt)
    edge_factor = smoothstep(abs(SOC - 50.0), 35.0, 90.0);
    power_factor = smoothstep(abs_power, 0.55, 1.15);
    rate_factor = smoothstep(abs(dSOC_dt), 12.0, 80.0);
    penalty = -36.0 * (edge_factor ^ 1.2) .* (power_factor ^ 1.1) .* (rate_factor ^ 0.9);

    recovery = 7.0 * exp(-(abs_power / 0.45) ^ 2) .* exp(-(abs(SOC - 50.0) / 28.0) ^ 2);

    value = bounded_value(penalty + recovery, -68.0, 12.0);
end

function value = compute_curiosity(P_batt_norm, SOC, dSOC_dt)
    modulation = sin(0.55 * P_batt_norm + 0.065 * SOC) ...
                 + 0.4 * sin(1.1 * P_batt_norm - 0.12 * SOC);
    envelope = exp(-(abs(SOC - 50.0) / 95.0) ^ 1.6) ...
               .* exp(-(abs(dSOC_dt) / 65.0) ^ 1.3);
    value = bounded_value(6.5 * modulation * envelope, -10.0, 10.0);
end

%% ------------------------------------------------------------------------
% Utility helpers
%% ------------------------------------------------------------------------

function y = smoothstep(x, edge0, edge1)
    if edge1 <= edge0
        y = double(x >= edge1);
        return;
    end
    t = bound_value((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    y = t .* t .* (3.0 - 2.0 * t);
end


function value = bound_value(value, lower, upper)
    if value < lower
        value = lower;
    elseif value > upper
        value = upper;
    end
end

function value = bounded_value(value, lower, upper)
    value = max(min(value, upper), lower);
end

function out = clamp_reward(value, neg_limit, pos_limit)
    out = value;
    if isnan(out) || isinf(out)
        out = 0.0;
        return;
    end
    out = max(-abs(neg_limit), min(pos_limit, out));
end

