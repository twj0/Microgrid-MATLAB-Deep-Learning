function P_out = P_manager(P_in, P_in_delay, SOC, SOC_delay, SOC_dot)
    %P_manager: 生成输出信号
    %输入：P_in，范围在[-1e4,1e4]
    %输入信号；P_in_delay，延迟后的输入信号
    %输出：P_out，输出信号
    %如果SOC在[0, 100]之外，输出的功率减少.

    P_in_clean = double(P_in);
    P_in_delay_clean = double(P_in_delay);
    SOC_clean = double(SOC);
    SOC_delay_clean = double(SOC_delay);

    original_size = size(P_in_clean);
    if isempty(original_size)
        original_size = [1 1];
    end
    P_vec = reshape(P_in_clean, [], 1);
    P_delay_vec = reshape(P_in_delay_clean, [], 1);
    P_out_vec = P_vec;

    if numel(P_vec) < 1
        P_out = zeros(original_size);
        return;
    end

    active_idx = 1;
    active_in = P_vec(active_idx);
    active_prev = P_delay_vec(min(active_idx, numel(P_delay_vec)));

    reactive_idx = 2;
    if numel(P_vec) >= reactive_idx
        reactive_in = P_vec(reactive_idx);
    else
        reactive_in = 0;
    end

    P_MAX = 80e3;
    RAMP_LIMIT = 4e3;
    SMOOTH_GAIN = 0.25;
    BRAKE_FACTOR = 0.3;
    SOC_EXP_SCALE = 8;
    SOC_DOT_DECAY = 1.5; % scaling for SOC derivative damping

    soc_now = SOC_clean(1);
    soc_prev = SOC_delay_clean(1);
    if nargin < 5 || isempty(SOC_dot)
        soc_dot_now = 0;
    else
        soc_dot_array = double(SOC_dot);
        if isempty(soc_dot_array)
            soc_dot_now = 0;
        else
            soc_dot_now = soc_dot_array(1);
        end
    end

    power_delta = active_in - active_prev;
    soc_delta = soc_now - soc_prev;

    below_limit = (soc_prev < 0) || (soc_now < 0);
    above_limit = (soc_prev > 100) || (soc_now > 100);

    if below_limit && (soc_delta < 0) && (power_delta < 0)
        active_in = active_prev + BRAKE_FACTOR * power_delta;
        power_delta = active_in - active_prev;
    elseif above_limit && (soc_delta > 0) && (power_delta > 0)
        active_in = active_prev + BRAKE_FACTOR * power_delta;
        power_delta = active_in - active_prev;
    end

    delta_active = clamp(power_delta, -RAMP_LIMIT, RAMP_LIMIT);
    filtered_active = active_prev + SMOOTH_GAIN * delta_active;

    soc_violation = 0;
    if soc_now < 0
        soc_violation = -soc_now;
    elseif soc_now > 100
        soc_violation = soc_now - 100;
    end
    if soc_violation > 0
        soc_scale = exp(-soc_violation / SOC_EXP_SCALE);
    else
        soc_scale = 1;
    end

    soc_trend = abs(soc_delta);
    trend_scale = clamp(1 - soc_trend / 30, 0.2, 1);

    deriv_scale = 1;
    if soc_now > 100 && soc_dot_now > 0
        deriv_scale = clamp(exp(-soc_dot_now / SOC_DOT_DECAY), 0.05, 1);
    elseif soc_now < 0 && soc_dot_now < 0
        deriv_scale = clamp(exp(soc_dot_now / SOC_DOT_DECAY), 0.05, 1);
    end

    active_out = filtered_active * soc_scale * trend_scale * deriv_scale;
    active_out = clamp(active_out, -P_MAX, P_MAX);

    reactive_out = clamp(reactive_in, -P_MAX, P_MAX);

    P_out_vec(active_idx) = active_out;
    if numel(P_out_vec) >= reactive_idx
        P_out_vec(reactive_idx) = reactive_out;
    end

    P_out = reshape(P_out_vec, original_size);
end

function y = clamp(x, lower, upper)
    %clamp: 将x限制在lower和upper之间
    y = min(max(x, lower), upper);
end
