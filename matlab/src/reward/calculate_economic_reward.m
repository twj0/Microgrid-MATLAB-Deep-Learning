function reward = calculate_economic_reward(P_net_load_W, P_batt_W, price, time_of_day)
%CALCULATE_ECONOMIC_REWARD 简化版经济奖励函数（仅购电场景 P_{grid}>0）
%   reward = calculate_economic_reward(P_net_load_W, P_batt_W, price, time_of_day)
%
%   设计（方案A：成本型 + 峰时宽容，平滑可微）：
%   - 仅考虑从电网购电（P_grid>0），忽略卖电；
%   - 低电价时“多买电”给正奖励，高电价时“少买电”给负奖励；
%   - 使用 softplus/tanh 实现渐进式、平滑饱和；
%   - 在人性化高峰时段（8~10, 18~22）对负向惩罚适度减弱（宽容因子）；
%   - 数值尺度：正向约≤+80，负向约≥-200；
%   - 输出限幅，防止极值扰动训练。
%
%   输入:
%     P_net_load_W  净负荷功率(W)
%     P_batt_W      电池功率(W)（>0充电，<0放电）
%     price         当前电价($/kWh)
%     time_of_day   当前小时[0,24]
%
%   说明:
%     本函数不含SOC门控；严格SOC约束在 Battery_Health_Penalty 或上层组合中实现。

% 基本防御
if ~isfinite(P_net_load_W), P_net_load_W = 0; end
if ~isfinite(P_batt_W),     P_batt_W     = 0; end
if ~isfinite(price),        price        = 0.45; end
if ~isfinite(time_of_day),  time_of_day  = 12; end

% 电网功率（仅购电有效）
P_grid_W  = P_net_load_W + P_batt_W;
P_grid_kW = max(P_grid_W, 0) / 1000.0;

% 价格归一（基于参考阈值）
PRICE_VALLEY = 0.391;
PRICE_PEAK   = 0.514;
if PRICE_PEAK <= PRICE_VALLEY
    z = 0.5; % 保底
else
    z = (price - PRICE_VALLEY) / (PRICE_PEAK - PRICE_VALLEY); % 0(谷)~1(峰)
    z = max(0.0, min(1.0, z));
end

% 功率归一与平滑购电量
P_REF_KW = 20.0;           % 参考功率（可按项目重标定）
p        = P_grid_kW / P_REF_KW;   % 无量纲
sp_k     = 3.0;                     % softplus斜率
softplus = @(x) log1p(exp(-abs(x))) + max(x, 0); % 数值稳定softplus
q        = softplus(sp_k * p) / sp_k;            % 平滑购电量

% 正向鼓励（低价多买）与负向惩罚（高价少买），均用tanh软饱和
A_POS   = 80.0;   % 正向上限标定
B_NEG   = 180.0;  % 负向上限标定（惩罚更强）
beta_q  = 5.0;    % 正向增长速率
gamma_q = 7.0;    % 负向增长速率（更陡）

reward_pos = A_POS * (1.0 - z) * tanh(beta_q  * q);
reward_neg = -B_NEG * z       * tanh(gamma_q * q);

% 人性化高峰时段宽容：仅减弱负向惩罚部分
is_human_peak = ((time_of_day >= 8  && time_of_day <= 10) || ...
                 (time_of_day >= 18 && time_of_day <= 22));
PEAK_RELIEF   = 0.30;  % 在人性化高峰时段减弱30%的惩罚
if is_human_peak
    reward_neg = (1.0 - PEAK_RELIEF) * reward_neg;
end

reward_raw = reward_pos + reward_neg;

% 限幅（防御NaN/Inf）
if ~isfinite(reward_raw), reward_raw = 0.0; end
reward = max(-200.0, min(80.0, reward_raw));
end
