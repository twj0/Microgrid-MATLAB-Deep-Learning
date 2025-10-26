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

% 新增：SOC高斯奖励参数与评论家缩放因子
SOC_GAUSS_TARGET    = 50;  % 目标SOC(%)
SOC_GAUSS_SIGMA     = 15;  % 高斯σ
CRITIC_SCALE        = 1.0; % 评论家输出基准缩放
CRITIC_RATIO_START  = 0.25; % gate 开启初期相对于基线的倍数
CRITIC_RATIO_TARGET = 1.50; % gate 全开后目标倍数
CRITIC_UNIT_REF     = 0.60; % 预期 critic_unit 平均值用于归一化

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
    grid_tol_pos = PGRID_TOL_CHARGE * tol_scale;    % 电网容差（谷时充电）
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

% 评分栈：加入方案A（时段-SOC匹配）与方案B（SOC目标轨迹）联合塑形
P_grid_kW = max(P_grid_eval, 0) / 1000.0;
P_REF_KW = 20.0;
q = min(P_grid_kW / P_REF_KW, 3.0);
% 价格归一（0=谷，1=峰）
z = (price - PRICE_VALLEY) / (PRICE_PEAK - PRICE_VALLEY);
if ~isfinite(z), z = 0.5; end
z = max(0.0, min(1.0, z));
% SOC门控（[20,80]内接近1，外快速衰减）
delta_soc = 3.0;
g_soc = 1.0 ./ (1.0 + exp(-(soc - 20.0) / delta_soc)) .* ...
        1.0 ./ (1.0 + exp(-(80.0 - soc) / delta_soc));
% 成本门控：电价×购电程度越高，越不利（越小越接近1）
g_cost = 1.0 ./ (1.0 + exp(((z .* tanh(5.0 * q)) - 0.35) / 0.10));

% 方案A：时段-SOC匹配度（谷低→充，峰高→放）
z0 = 0.35; z1 = 0.65; dz = 0.10; SOC_low_thr = 40; SOC_high_thr = 60; delta_thr = 3.0;
s_valley = 1.0 ./ (1.0 + exp((z - z0) / dz));      % 越接近谷越接近1
s_peak   = 1.0 ./ (1.0 + exp((z1 - z) / dz));      % 越接近峰越接近1
m_valley = s_valley .* 1.0 ./ (1.0 + exp((soc - SOC_low_thr) / delta_thr));   % 低SOC在谷时更优
m_peak   = s_peak   .* 1.0 ./ (1.0 + exp((SOC_high_thr - soc) / delta_thr)); % 高SOC在峰时更优
m = 0.5 * m_valley + 0.5 * m_peak;

% 方案B：SOC目标轨迹（TOU参考曲线）
SOCv = 65; SOCf = 50; SOCp = 35; sigma = 10;
w_val  = 1.0 ./ (1.0 + exp((z - z0) / dz));
w_peak = 1.0 ./ (1.0 + exp((z1 - z) / dz));
w_flat = max(0.0, 1.0 - w_val - w_peak);
wnorm  = max(1e-6, (w_val + w_flat + w_peak));
SOC_ref = (w_val.*SOCv + w_flat.*SOCf + w_peak.*SOCp) ./ wnorm;
r_track = exp(-((soc - SOC_ref).^2) ./ (2.0 * sigma^2));

% 组合评分栈（权重按确认方案），新增项乘以gate渐进启用
w_batt = 0.12;
w_grid = 0.12;
w_balance = 0.12;
w_soc = 0.14;
w_cost = 0.14;
w_soc_gauss = 0.10;
w_smooth = 0.08;
w_sync = 0.06;
w_match = 0.06;
w_track = 0.06;
% 使用局部门控 gate_local 只渐进启用新增项（总体评论家仍有全局 gate）
edge0 = 500; edge1 = 1000;
if edge1 <= edge0
    gate_local = double(episode_num >= edge1);
else
    t_local = max(0.0, min(1.0, (episode_num - edge0) / (edge1 - edge0)));
    gate_local = t_local .* t_local .* (3.0 - 2.0 * t_local);
end

stacked_score = ...
    w_batt * r_batt + ...
    w_grid * r_grid + ...
    w_balance * r_balance + ...
    w_soc * g_soc + ...
    w_cost * g_cost + ...
    w_soc_gauss * soc_gauss + ...
    w_smooth * r_smooth + ...
    w_sync * r_time_sync + ...
    w_match * (gate_local .* m) + ...
    w_track * (gate_local .* r_track);

s = stacked_score;
critic_unit = saturate_unit(s);

% Episode9718454696362655500	658769000 541 5675952)
edge0 = 500; edge1 = 1000;
if edge1 <= edge0
    gate = double(episode_num >= edge1);
else
    t = max(0.0, min(1.0, (episode_num - edge0) / (edge1 - edge0)));
    gate = t .* t .* (3.0 - 2.0 * t);
end

% 防御性：对NaN/Inf输入做降级处理，避免奖励传播出错
if ~isfinite(econ_reward), econ_reward = 0; end
if ~isfinite(health_penalty), health_penalty = 0; end
pos_econ = max(econ_reward, 0);
pos_health = max(health_penalty, 0);

baseline_reward = pos_econ + pos_health;
if baseline_reward < 1e-6
    baseline_reward = 1e-6;
end

critic_ratio = CRITIC_RATIO_START + (CRITIC_RATIO_TARGET - CRITIC_RATIO_START) * gate;
alpha_eff = baseline_reward * critic_ratio / max(CRITIC_UNIT_REF, 1e-3);

critic_reward = CRITIC_SCALE * gate * alpha_eff * critic_unit;

% 组合原始奖励并进行软/硬裁剪，抑制极端值（避免EpisodeQ0暴发到极大负数/正数）
raw_total = pos_econ + pos_health + critic_reward;
soft_total = REWARD_SOFTCLIP * tanh(raw_total / REWARD_SOFTCLIP);  % tanh 压缩（软裁剪）
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
