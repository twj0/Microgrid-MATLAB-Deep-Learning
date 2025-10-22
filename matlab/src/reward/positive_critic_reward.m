function [total_reward, critic_unit, gate] = positive_critic_reward(econ_reward, health_penalty, soc, time_hour, P_batt_W, price, P_net_load_W, P_grid_W, episode_num) %#codegen
% 改动：加入SOC高斯奖励 + 评论家输出缩放（默认0.5）
PRICE_VALLEY = 0.391;
PRICE_PEAK = 0.514;
SOC_CENTER = 50;
SOC_CENTER_WIDTH = 25;       % widened center width to reduce sparsity
SOC_BAND_LOW = 25;           % expanded band
SOC_BAND_HIGH = 85;
SOC_BAND_MARGIN = 15;        % softer margin outside band
PBATT_TARGET_VALLEY = 6000;
PBATT_TARGET_PEAK = -6000;
PBATT_TARGET_FLAT = 0;
PBATT_TOL_CHARGE = 6000;     % looser tolerance
PBATT_TOL_DISCHARGE = 6000;  % looser tolerance
PBATT_TOL_IDLE = 5000;       % looser tolerance for flat
PGRID_TARGET_VALLEY = 5000;
PGRID_TARGET_PEAK = -5000;
PGRID_TARGET_FLAT = 0;
PGRID_TOL_CHARGE = 8000;     % looser grid tolerance
PGRID_TOL_DISCHARGE = 8000;
PGRID_TOL_IDLE = 6000;
SMOOTH_REF = 6000;           % smoothing threshold (used with deltaP)
NET_BAL_TOL = 8000;          % slightly looser net balance
SWITCH_EP = 450;
RAMP_WINDOW = 100;
ALPHA = 3000;                % base scaling (will be staged)
RAMP_UP_END = 800;           % 前500-1000个episode更平滑的权重提升（此处取800）
STABLE_START = 1600;         % 1600之后进入收敛稳定期（最后400个episode保持不变）
ALPHA_MAX = 9000;            % 目标权重上限（目标值）

% 新增：SOC高斯奖励参数与评论家缩放因子
SOC_GAUSS_TARGET = 50;  % 目标SOC(%)
SOC_GAUSS_SIGMA  = 15;  % 高斯σ
CRITIC_SCALE     = 0.3; % 评论家输出整体缩放到原来的30%（如需更小量级可调至0.1）

% 课程式容差放宽与奖励裁剪参数（使早期更易学，逐步收敛）
TOL_EASE_END = 800;           % 前800个episode逐步从宽容到严格
TOL_EASE_MAX_SCALE = 1.6;     % 早期容差放宽比例（例如1.6倍）
REWARD_SOFTCLIP = 1e5;        % 奖励软裁剪尺度（tanh压缩）
REWARD_HARDCLIP = 3e5;        % 奖励硬裁剪上限，避免极端值

% 基于回合号的阶段参数（供容差/权重调度）
ep = max(episode_num, 1);
phase_tol = min(ep / TOL_EASE_END, 1);
% 早期容差更宽松，后期回归至1.0
			 tol_scale = TOL_EASE_MAX_SCALE - (TOL_EASE_MAX_SCALE - 1) * phase_tol;

[valley, peak, flat, time_valley, time_peak, time_flat] = classify_zone(price, time_hour, PRICE_VALLEY, PRICE_PEAK);

if ~isfinite(P_grid_W)
    P_grid_eval = P_net_load_W + P_batt_W;
else
    P_grid_eval = P_grid_W;
end

r_soc_center = saturate_unit(1 - abs(soc - SOC_CENTER) / SOC_CENTER_WIDTH);
if soc < SOC_BAND_LOW
    r_soc_band = saturate_unit(1 - (SOC_BAND_LOW - soc) / SOC_BAND_MARGIN);
elseif soc > SOC_BAND_HIGH
    r_soc_band = saturate_unit(1 - (soc - SOC_BAND_HIGH) / SOC_BAND_MARGIN);
else
    r_soc_band = saturate_unit(1 - abs(soc - SOC_CENTER) / ((SOC_BAND_HIGH - SOC_BAND_LOW) * 0.5));
end
r_soc = saturate_unit(0.65 * r_soc_center + 0.35 * r_soc_band);

if valley
    target_batt = PBATT_TARGET_VALLEY;
    tol_pos = PBATT_TOL_CHARGE * tol_scale;    % 课程式：早期容差更宽松
    tol_neg = PBATT_TOL_DISCHARGE * tol_scale;
elseif peak
    target_batt = PBATT_TARGET_PEAK;
    tol_pos = PBATT_TOL_CHARGE * tol_scale;
    tol_neg = PBATT_TOL_DISCHARGE * tol_scale;
else
    target_batt = PBATT_TARGET_FLAT;
    tol_pos = PBATT_TOL_IDLE * tol_scale;
    tol_neg = PBATT_TOL_IDLE * tol_scale;
end
r_batt = triangular_score(P_batt_W, target_batt, tol_pos, tol_neg);

if valley
    target_grid = PGRID_TARGET_VALLEY;
    grid_tol_pos = PGRID_TOL_CHARGE * tol_scale;    % 
    grid_tol_neg = PGRID_TOL_DISCHARGE * tol_scale;
elseif peak
    target_grid = PGRID_TARGET_PEAK;
    grid_tol_pos = PGRID_TOL_CHARGE * tol_scale;
    grid_tol_neg = PGRID_TOL_DISCHARGE * tol_scale;
else
    target_grid = PGRID_TARGET_FLAT;
    grid_tol_pos = PGRID_TOL_IDLE * tol_scale;
    grid_tol_neg = PGRID_TOL_IDLE * tol_scale;
end
r_grid = triangular_score(P_grid_eval, target_grid, grid_tol_pos, grid_tol_neg);

net_balance = P_net_load_W + P_batt_W - P_grid_eval;
r_balance = saturate_unit(1 - abs(net_balance) / NET_BAL_TOL);

r_time_sync = double((valley && time_valley) || (peak && time_peak) || (flat && time_flat));

% Smoothness relative to desired operating span around target power
if valley || peak
    target_span = PBATT_TOL_CHARGE;
else
    target_span = PBATT_TOL_IDLE;
end
dP = abs(P_batt_W - target_batt);
if target_span <= 0
    r_smooth = double(dP == 0);
else
    smooth_scale = target_span + SMOOTH_REF;
    r_smooth = saturate_unit(1 - dP / smooth_scale);
end

% 新增：SOC高斯奖励项
soc_gauss = exp(-((soc - SOC_GAUSS_TARGET)^2) / (2 * SOC_GAUSS_SIGMA^2));

% 重新分配权重：原六项整体×0.8，预留0.2给SOC高斯
critic_unit = ...
    0.224 * r_soc     + ...
    0.200 * r_batt    + ...
    0.144 * r_grid    + ...
    0.096 * r_balance + ...
    0.080 * r_smooth  + ...
    0.056 * r_time_sync + ...
    0.200 * soc_gauss;
critic_unit = saturate_unit(critic_unit);

gate = (episode_num - SWITCH_EP) / RAMP_WINDOW;
gate = saturate_unit(gate);

% 评论家权重调度：
% - 前800个episode线性提升：从初始ALPHA平滑增长到ALPHA_MAX（让前500-1000个episode更渐进）
% - 800<episode<1600：保持权重不变（稳定阶段）
% - episode>=1600：固定不变，展示收敛走势
ep = max(episode_num, 1);
if ep <= RAMP_UP_END
    ramp_ratio = ep / RAMP_UP_END; % [0,1]
    alpha_eff = ALPHA + (ALPHA_MAX - ALPHA) * ramp_ratio;
elseif ep < STABLE_START
    alpha_eff = ALPHA_MAX;
else
    alpha_eff = ALPHA_MAX;
end

% 防御性：对NaN/Inf输入做降级处理，避免奖励传播出错
if ~isfinite(econ_reward), econ_reward = 0; end
if ~isfinite(health_penalty), health_penalty = 0; end
pos_econ = max(econ_reward, 0);
pos_health = max(health_penalty, 0);
critic_reward = CRITIC_SCALE * gate * alpha_eff * critic_unit;

% 组合原始奖励并进行软/硬裁剪，抑制极端值（避免EpisodeQ0暴发到极大负数/正数）
raw_total = pos_econ + pos_health + critic_reward;
soft_total = REWARD_SOFTCLIP * tanh(raw_total / REWARD_SOFTCLIP);  % tanh
% 硬裁剪保护
 total_reward = max(min(soft_total, REWARD_HARDCLIP), -REWARD_HARDCLIP);

% 返回前有效性检查：若出现NaN/Inf，给出警告并返回大负值以稳定训练
if ~isfinite(total_reward)
    warning('positive_critic_reward:InvalidReward', ...
        'Reward invalid: econ=%0.3g, health=%0.3g, critic=%0.3g, ep=%d', ...
        econ_reward, health_penalty, critic_reward, episode_num);
    total_reward = -1e6;
end
end

function [valley, peak, flat, time_valley, time_peak, time_flat] = classify_zone(price, time_hour, price_valley, price_peak)
time_valley = (time_hour < 7) || (time_hour >= 23);
time_peak = (time_hour >= 11 && time_hour < 14) || (time_hour >= 18 && time_hour < 22);
time_flat = ~(time_valley || time_peak);
if isfinite(price)
    valley = price <= price_valley;
    peak = price >= price_peak;
    flat = ~(valley || peak);
    if ~valley && time_valley
        valley = true;
        peak = false;
        flat = false;
    elseif ~peak && time_peak
        peak = true;
        valley = false;
        flat = false;
    elseif ~(valley || peak)
        flat = true;
    end
else
    valley = time_valley;
    peak = time_peak;
    flat = time_flat;
end
end

function score = triangular_score(value, target, width_pos, width_neg)
diff_val = value - target;
if diff_val >= 0
    width = width_pos;
else
    width = width_neg;
end
if width <= 0
    score = double(diff_val == 0);
else
    score = 1 - abs(diff_val) / width;
end
score = saturate_unit(score);
end

function y = saturate_unit(x)
if x < 0
    y = 0;
elseif x > 1
    y = 1;
else
    y = x;
end
end
