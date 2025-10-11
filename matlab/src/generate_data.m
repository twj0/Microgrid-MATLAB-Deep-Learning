%% 微电网仿真数据生成脚本
% =========================================================================
% 功能: 生成光伏出力、可变负载和电价曲线数据
% 用途: 为Simulink微电网模型提供输入数据
% 时间分辨率: 1小时
% 仿真时长: 30天 (720小时)
% =========================================================================

clc; clear; close all;

fprintf('=== 微电网仿真数据生成 ===\n');

%% 1. 基本参数设置
% =========================================================================
simulation_days = 30;           % 仿真天数
hours_per_day = 24;             % 每天小时数
total_hours = simulation_days * hours_per_day;  % 总小时数 = 720
dt = 3600;                      % 时间步长(秒) = 1小时
time_hours = (0:total_hours-1)';    % 时间向量(小时)
time_seconds = time_hours * dt;     % 时间向量(秒)

fprintf('仿真参数:\n');
fprintf('  - 仿真天数: %d 天\n', simulation_days);
fprintf('  - 总小时数: %d 小时\n', total_hours);
fprintf('  - 时间步长: %d 秒 (1小时)\n', dt);

%% 2. 光伏出力曲线生成
% =========================================================================
fprintf('\n正在生成光伏出力曲线...\n');

% 光伏系统参数
pv_rated_power = 100e3;         % 光伏额定功率 100 kW
sunrise_hour = 6;               % 日出时刻
sunset_hour = 18;               % 日落时刻
peak_hour = 12;                 % 峰值时刻(正午)

% 初始化光伏功率数组
pv_power = zeros(total_hours, 1);

% 为每一天生成光伏曲线
for day = 0:(simulation_days-1)
    % 随机生成天气因子 (0.3-1.0)
    % 0.3 = 阴天, 1.0 = 晴天
    weather_factor = 0.3 + 0.7 * rand();
    
    % 添加随机波动
    cloud_noise = 0.1 * randn();  % 云层扰动
    weather_factor = max(0.2, min(1.0, weather_factor + cloud_noise));
    
    % 生成该天的24小时数据
    for hour = 0:23
        time_index = day * hours_per_day + hour + 1;
        hour_of_day = hour;
        
        % 日间时段 (6:00-18:00)
        if hour_of_day >= sunrise_hour && hour_of_day < sunset_hour
            % 使用修正的正弦函数模拟太阳辐照
            % 峰值在正午12点
            time_from_sunrise = hour_of_day - sunrise_hour;
            daylight_hours = sunset_hour - sunrise_hour;
            normalized_time = time_from_sunrise / daylight_hours;
            
            % 贝塔分布更符合实际光伏曲线 (早晚低, 中午高)
            % 这里用正弦函数的平方近似
            solar_profile = sin(pi * normalized_time)^2;
            
            % 添加小幅随机波动 (±5%)
            hourly_noise = 1 + 0.05 * (2*rand() - 1);
            
            % 计算该小时的光伏出力
            pv_power(time_index) = pv_rated_power * solar_profile * ...
                                   weather_factor * hourly_noise;
        else
            % 夜间无出力
            pv_power(time_index) = 0;
        end
    end
end

% 确保功率非负
pv_power = max(0, pv_power);

fprintf('  - 光伏额定功率: %.1f kW\n', pv_rated_power/1000);
fprintf('  - 平均日发电量: %.2f kWh\n', mean(sum(reshape(pv_power(1:simulation_days*24), 24, []), 1))/1000);
fprintf('  - 最大瞬时功率: %.2f kW\n', max(pv_power)/1000);

%% 3. 可变负载曲线生成
% =========================================================================
fprintf('\n正在生成可变负载曲线...\n');

% 负载参数 (商业综合体/小区场景)
base_load = 25e3;               % 基础负载 25 kW (固定负载)
peak_load_multiplier = 2.5;     % 峰值倍数
weekend_reduction = 0.7;        % 周末负载降低系数

% 负载峰值时段定义
morning_peak_start = 8;         % 早高峰开始
morning_peak_end = 11;          % 早高峰结束
evening_peak_start = 18;        % 晚高峰开始
evening_peak_end = 22;          % 晚高峰结束
valley_start = 23;              % 低谷开始
valley_end = 7;                 % 低谷结束

% 初始化负载功率数组
load_power = zeros(total_hours, 1);

% 为每一天生成负载曲线
for day = 0:(simulation_days-1)
    % 判断是否为周末 (简化: 每7天中第6、7天为周末)
    is_weekend = (mod(day, 7) >= 5);
    day_factor = 1.0;
    if is_weekend
        day_factor = weekend_reduction;
    end
    
    % 生成该天的24小时数据
    for hour = 0:23
        time_index = day * hours_per_day + hour + 1;
        hour_of_day = hour;
        
        % 基础负载曲线 (使用多个高斯分布叠加)
        % 早高峰 (8:00-11:00)
        morning_peak_profile = exp(-((hour_of_day - 9.5)^2) / (2 * 1.5^2));
        
        % 晚高峰 (18:00-22:00)
        evening_peak_profile = exp(-((hour_of_day - 20)^2) / (2 * 2^2));
        
        % 午间负载 (12:00-14:00)
        noon_profile = 0.5 * exp(-((hour_of_day - 13)^2) / (2 * 1^2));
        
        % 夜间低谷
        if hour_of_day >= valley_start || hour_of_day <= valley_end
            valley_factor = 0.4;  % 夜间负载降至40%
        else
            valley_factor = 1.0;
        end
        
        % 综合负载系数 (0.4-2.5之间)
        load_factor = valley_factor + ...
                      (peak_load_multiplier - valley_factor) * ...
                      (morning_peak_profile + evening_peak_profile + noon_profile);
        
        % 限制在合理范围
        load_factor = max(0.4, min(peak_load_multiplier, load_factor));
        
        % 添加随机波动 (±10%)
        random_fluctuation = 1 + 0.1 * (2*rand() - 1);
        
        % 计算该小时的负载功率
        load_power(time_index) = base_load * load_factor * ...
                                 day_factor * random_fluctuation;
    end
end

% 确保负载为正
load_power = max(base_load * 0.3, load_power);

fprintf('  - 基础负载: %.1f kW\n', base_load/1000);
fprintf('  - 平均负载: %.2f kW\n', mean(load_power)/1000);
fprintf('  - 峰值负载: %.2f kW\n', max(load_power)/1000);
fprintf('  - 平均日用电量: %.2f kWh\n', mean(sum(reshape(load_power(1:simulation_days*24), 24, []), 1))/1000);

%% 4. 分时电价曲线生成 (夏季E-TOU-C费率)
% =========================================================================
fprintf('\n正在生成分时电价曲线...\n');

% 电价参数 ($/kWh) - PG&E夏季E-TOU-C费率 (普通用户Tier 1)
price_off_peak = 0.39;  % 非高峰时段电价 (39¢/kWh)
price_peak = 0.51;      % 高峰时段电价 (51¢/kWh)

% 高峰时段定义: 4 p.m. - 9 p.m. (16:00-21:00)
peak_start_hour = 16;   % 高峰开始时间
peak_end_hour = 21;     % 高峰结束时间

% 初始化电价数组
price_array = zeros(total_hours, 1);

% 为每个小时分配电价
for i = 1:total_hours
    hour_of_day = mod(time_hours(i), 24);
    
    % 判断是否在高峰时段 (4-9 PM)
    if hour_of_day >= peak_start_hour && hour_of_day < peak_end_hour
        price_array(i) = price_peak;      % 高峰时段
    else
        price_array(i) = price_off_peak;  % 非高峰时段
    end
end

% 验证电价数据
price_min = min(price_array);
price_max = max(price_array);

fprintf('  - 费率方案: PG&E E-TOU-C (夏季, 普通用户)\n');
fprintf('  - 高峰时段: %d:00 - %d:00\n', peak_start_hour, peak_end_hour);
fprintf('  - 高峰电价: $%.3f/kWh (51¢/kWh)\n', price_peak);
fprintf('  - 非高峰电价: $%.3f/kWh (39¢/kWh)\n', price_off_peak);
fprintf('  - 电价范围: $%.3f ~ $%.3f/kWh\n', price_min, price_max);

%% 5. 电池特性查找表 (Look-up Tables)
% =========================================================================
% 此步骤已停用，根据最新需求不再生成电池查找表或相关图像。
%{
fprintf('\n正在创建电池特性查找表...\n');

% 电池系统参数
cell_capacity_Ah = 280;         % 单电芯容量 280 Ah (EVE LF280K)
num_cells_series = 250;         % 串联电芯数
system_capacity_Ah = cell_capacity_Ah;  % 系统容量 (250S1P配置)
system_voltage_nominal = 800;   % 系统标称电压 800V

fprintf('  - 电芯型号: EVE LF280K (LFP)\n');
fprintf('  - 单芯容量: %d Ah\n', cell_capacity_Ah);
fprintf('  - 系统配置: %dS1P\n', num_cells_series);
fprintf('  - 系统容量: %d Ah (%.1f kWh)\n', system_capacity_Ah, system_capacity_Ah*system_voltage_nominal/1000);

% 5.1. SOC-OCV 查找表 (开路电压)
% 基于EVE LF280K在25°C下的实测数据
% 参考: 电池建模文档 - SOC-OCV数据表
fprintf('\n  正在生成 SOC-OCV 查找表...\n');

% SOC断点 (%)
SOC_breakpoints = [0,  1,  5,   10,  20,  30,  40,  50,  60,  70,  80,  90,  95,  99, 100];

% OCV值 (V) - 单电芯电压
% LFP特性: 20%-90%区间平坦, 低SOC快速下降
OCV_values = [2.50, 2.73, 2.90, 3.00, 3.20, 3.22, 3.25, 3.26, 3.27, 3.30, 3.32, 3.35, 3.40, 3.50, 3.65];

% 创建Simulink.LookupTable对象 (推荐做法)
OCV_LUT = Simulink.LookupTable;
OCV_LUT.Breakpoints.Value = SOC_breakpoints;
OCV_LUT.Table.Value = OCV_values;

fprintf('    - SOC范围: %.0f%% - %.0f%%\n', min(SOC_breakpoints), max(SOC_breakpoints));
fprintf('    - OCV范围: %.2f V - %.2f V (单电芯)\n', min(OCV_values), max(OCV_values));
fprintf('    - 数据点数: %d\n', length(SOC_breakpoints));

% 5.2. SOC-R_int 查找表 (直流内阻)
% 基于典型LFP电池DCIR曲线估计
% 参考: 电池建模文档 - SOC-R_int数据表
fprintf('\n  正在生成 SOC-R_int 查找表...\n');

% R_int值 (mΩ) - 单电芯内阻
% U型曲线: 中间低(0.18 mΩ@50-60%SOC), 两端高(0.60 mΩ@0%SOC)
R_int_values_mOhm = [0.60, 0.50, 0.40, 0.45, 0.35, 0.25, 0.20, 0.18, 0.18, 0.19, 0.20, 0.22, 0.25, 0.35, 0.40];

% 转换为欧姆 (Ω)
R_int_values_Ohm = R_int_values_mOhm / 1000;

% 创建Simulink.LookupTable对象
R_int_LUT = Simulink.LookupTable;
R_int_LUT.Breakpoints.Value = SOC_breakpoints;
R_int_LUT.Table.Value = R_int_values_Ohm;

fprintf('    - R_int范围: %.2f mΩ - %.2f mΩ (单电芯)\n', min(R_int_values_mOhm), max(R_int_values_mOhm));
fprintf('    - 最低内阻: %.2f mΩ @ 50-60%% SOC\n', min(R_int_values_mOhm));
fprintf('    - 数据点数: %d\n', length(SOC_breakpoints));

% 5.3. 可视化电池特性曲线
fprintf('\n  正在绘制电池特性曲线...\n');

figure('Name', '电池特性查找表', 'Position', [100, 100, 1200, 500]);

% 子图1: SOC-OCV曲线
subplot(1,2,1);
plot(SOC_breakpoints, OCV_values, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
xlabel('SOC (%)');
ylabel('开路电压 OCV (V)');
title('EVE LF280K - SOC vs OCV 特性曲线 (25°C)');
grid on;
xlim([0, 100]);
ylim([2.4, 3.7]);
% 标注平台区
hold on;
fill([20, 90, 90, 20], [3.15, 3.15, 3.38, 3.38], 'y', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
text(55, 3.42, '平台区 (LFP特性)', 'HorizontalAlignment', 'center', 'FontSize', 10);
hold off;

% 子图2: SOC-R_int曲线
subplot(1,2,2);
plot(SOC_breakpoints, R_int_values_mOhm, 'r-s', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
xlabel('SOC (%)');
ylabel('直流内阻 R_{int} (mΩ)');
title('EVE LF280K - SOC vs R_{int} 特性曲线 (25°C)');
grid on;
xlim([0, 100]);
ylim([0, max(R_int_values_mOhm)*1.1]);
% 标注最低点
hold on;
[min_R, min_idx] = min(R_int_values_mOhm);
plot(SOC_breakpoints(min_idx), min_R, 'go', 'MarkerSize', 10, 'LineWidth', 2);
text(SOC_breakpoints(min_idx), min_R-0.05, sprintf('最低: %.2f mΩ', min_R), ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 9);
hold off;
%}

%% 6. 创建Timeseries对象 (供Simulink使用)
% =========================================================================
fprintf('\n正在创建Timeseries对象...\n');

% 光伏出力时间序列
pv_power_profile = timeseries(pv_power, time_seconds);
pv_power_profile.Name = 'PV Power Profile';
pv_power_profile.DataInfo.Units = 'W';

% 可变负载时间序列
load_power_profile = timeseries(load_power, time_seconds);
load_power_profile.Name = 'Load Power Profile';
load_power_profile.DataInfo.Units = 'W';

% 电价时间序列 (改名为price_profile以匹配Simulink)
price_data = price_array;  % 保存原始数组
price_profile = timeseries(price_data, time_seconds);
price_profile.Name = 'Electricity Price Profile';
price_profile.DataInfo.Units = 'CNY/kWh';

fprintf('  - pv_power_profile: %d 个数据点\n', length(pv_power));
fprintf('  - load_power_profile: %d 个数据点\n', length(load_power));
fprintf('  - price_profile: %d 个数据点\n', length(price_data));

%% 7. 保存数据到工作区和.mat文件
% =========================================================================
fprintf('\n正在保存数据...\n');

% 保存到.mat文件
save_filename = 'microgrid_simulation_data.mat';
save(save_filename, 'pv_power_profile', 'load_power_profile', 'price_profile', ...
     'price_data', 'time_hours', 'time_seconds', 'simulation_days', 'dt');

fprintf('  - 数据已保存到: %s\n', save_filename);
fprintf('  - 包含内容: 光伏/负载/电价曲线\n');

%% 8. 数据可视化
% =========================================================================
fprintf('\n正在绘制数据曲线...\n');

% 创建图形窗口
figure('Name', '微电网仿真输入数据', 'Position', [100, 100, 1400, 800]);

% 可视化全部30天的数据
vis_days = simulation_days;  % 显示全部30天
vis_hours = vis_days * 24;
time_vis = time_hours(1:vis_hours);

% 子图1: 光伏出力
subplot(3,1,1);
plot(time_vis, pv_power(1:vis_hours)/1000, 'b-', 'LineWidth', 1.5);
xlabel('时间 (小时)');
ylabel('光伏出力 (kW)');
title(sprintf('光伏发电功率曲线 (%d天)', vis_days));
grid on;
xlim([0, vis_hours]);
ylim([0, max(pv_power)/1000*1.1]);

% 子图2: 负载功率
subplot(3,1,2);
plot(time_vis, load_power(1:vis_hours)/1000, 'r-', 'LineWidth', 1.5);
xlabel('时间 (小时)');
ylabel('负载功率 (kW)');
title(sprintf('负载功率曲线 (%d天)', vis_days));
grid on;
xlim([0, vis_hours]);
ylim([0, max(load_power)/1000*1.1]);

% 子图3: 电价
subplot(3,1,3);
stairs(time_vis, price_data(1:vis_hours), 'k-', 'LineWidth', 1.5);
xlabel('时间 (小时)');
ylabel('电价 ($/kWh)');
title(sprintf('PG&E E-TOU-C夏季分时电价曲线 (%d天)', vis_days));
grid on;
xlim([0, vis_hours]);
ylim([0, max(price_data)*1.2]);

% 添加高峰和非高峰时段标注
hold on;
yline(price_peak, '--r', sprintf('高峰时段(%d--%d PM) $%.3f/kWh', peak_start_hour-12, peak_end_hour-12), ...
      'LineWidth', 1, 'LabelHorizontalAlignment', 'left');
yline(price_off_peak, '--b', sprintf('非高峰时段 $%.3f/kWh', price_off_peak), ...
      'LineWidth', 1, 'LabelHorizontalAlignment', 'right');
hold off;

%% 9. 统计分析
% =========================================================================
fprintf('\n=== 数据统计分析 ===\n');

% 光伏统计
pv_total_energy = sum(pv_power) / 1000;  % kWh
pv_avg_power = mean(pv_power) / 1000;    % kW
fprintf('\n光伏系统:\n');
fprintf('  - 总发电量: %.2f kWh (%d天)\n', pv_total_energy, simulation_days);
fprintf('  - 平均功率: %.2f kW\n', pv_avg_power);
fprintf('  - 日均发电量: %.2f kWh/天\n', pv_total_energy/simulation_days);

% 负载统计
load_total_energy = sum(load_power) / 1000;  % kWh
load_avg_power = mean(load_power) / 1000;    % kW
fprintf('\n负载系统:\n');
fprintf('  - 总用电量: %.2f kWh (%d天)\n', load_total_energy, simulation_days);
fprintf('  - 平均功率: %.2f kW\n', load_avg_power);
fprintf('  - 日均用电量: %.2f kWh/天\n', load_total_energy/simulation_days);

% 净负荷统计
net_load = load_power - pv_power;
net_load_energy = sum(net_load) / 1000;  % kWh
fprintf('\n净负荷 (负载-光伏):\n');
fprintf('  - 净用电量: %.2f kWh\n', net_load_energy);
fprintf('  - 光伏自用率: %.2f%%\n', (1 - net_load_energy/load_total_energy)*100);

% 无储能情况下的电费估算
cost_no_storage = 0;
for i = 1:total_hours
    if net_load(i) > 0  % 从电网购电
        cost_no_storage = cost_no_storage + (net_load(i)/1000) * price_data(i);
    end
end
fprintf('\n无储能情况下的电费估算:\n');
fprintf('  - 总电费: %.2f 元 (%d天)\n', cost_no_storage, simulation_days);
fprintf('  - 日均电费: %.2f 元/天\n', cost_no_storage/simulation_days);

fprintf('\n=== 数据生成完成 ===\n');
fprintf('变量已加载到工作区，可直接用于Simulink仿真。\n');

