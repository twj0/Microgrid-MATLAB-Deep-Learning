%% 微电网DDPG强化学习 - 主程序
% =========================================================================
% 功能: 训练DDPG智能体优化微电网储能调度
% 模型: Microgrid.slx
% 算法: Deep Deterministic Policy Gradient (DDPG)
% =========================================================================

clc; clear; close all;

fprintf('========================================\n');
fprintf('  微电网DDPG强化学习训练系统\n');
fprintf('========================================\n');
fprintf('启动时间: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf('========================================\n\n');

%% 1. 加载仿真数据
% =========================================================================
fprintf('=== 步骤1: 加载仿真数据 ===\n');

% 数据文件路径
data_path = fullfile('..', 'src', 'microgrid_simulation_data.mat');

if ~exist(data_path, 'file')
    error('数据文件不存在: %s\n请先运行 matlab/src/generate_data.m', data_path);
end

% 加载数据
load(data_path);
fprintf('✓ 数据加载成功\n');

% 兼容旧版数据: 确保 price_profile 存在且为 timeseries
price_reconstructed = false;
if ~exist('price_profile', 'var') || isempty(price_profile)
    if exist('price_data', 'var') && ~isempty(price_data)
        price_profile = price_data;
        price_reconstructed = true;
    else
        error(['数据缺失: 未找到 price_profile 或 price_data。' ...
               '请重新运行 matlab/src/generate_data.m 生成完整数据。']);
    end
end

if isa(price_profile, 'timeseries')
    % 若时间向量缺失, 使用已知时间基准补全
    if isempty(price_profile.Time) && exist('time_seconds', 'var')
        sample_count = getTimeseriesSampleCount(price_profile);
        if sample_count == numel(time_seconds)
            price_profile.Time = time_seconds(:);
        end
    end
elseif isnumeric(price_profile)
    % 将数值数组转换为timeseries
    if exist('time_seconds', 'var') && numel(time_seconds) == numel(price_profile)
        time_vector = time_seconds(:);
    else
        if exist('dt', 'var') && isnumeric(dt) && isscalar(dt)
            dt_seconds = double(dt);
        else
            dt_seconds = 3600;  % 默认1小时步长
        end
        time_vector = (0:numel(price_profile)-1)' * dt_seconds;
    end
    price_profile = timeseries(price_profile(:), time_vector);
    price_reconstructed = true;
else
    error('数据格式错误: 无法识别 price_profile 类型 %s', class(price_profile));
end

if price_reconstructed
    price_profile.Name = 'Electricity Price Profile';
    price_profile.DataInfo.Units = 'CNY/kWh';
    fprintf('  - 已自动重建电价时间序列 (price_profile)\n');
end

% 检查数据格式和长度
if ~isa(pv_power_profile, 'timeseries') || ~isa(load_power_profile, 'timeseries')
    error('数据格式错误: pv_power_profile 不是 timeseries 对象');
end

pv_hours = getTimeseriesDurationHours(pv_power_profile);
load_hours = getTimeseriesDurationHours(load_power_profile);
price_hours = getTimeseriesDurationHours(price_profile);

fprintf('  - 光伏数据: %.0f 小时\n', pv_hours);
fprintf('  - 负载数据: %.0f 小时\n', load_hours);
fprintf('  - 电价数据: %.0f 小时\n', price_hours);

if exist('OCV_LUT', 'var') && exist('R_int_LUT', 'var')
    fprintf('  - 电池查找表: OCV_LUT, R_int_LUT\n');
else
    fprintf('  - 警告: 电池查找表不存在，请重新生成数据\n');
end

% 检查数据长度
if pv_hours < 700
    fprintf('\n⚠️  警告: 数据长度不足！\n');
    fprintf('  当前数据: %.0f 小时\n', pv_hours);
    fprintf('  期望数据: 720 小时 (30天)\n');
    fprintf('  请运行以下命令重新生成数据:\n');
    fprintf('    cd matlab/src\n');
    fprintf('    run(''generate_data.m'')\n\n');
    error('数据长度不足，无法继续训练');
end

agentVarName = '';

%% 2. 配置Simulink模型
% =========================================================================
fprintf('\n=== 步骤2: 配置Simulink模型 ===\n');

model_name = 'Microgrid';
model_dir = fullfile('..', '..', 'model');
model_path = fullfile(model_dir, [model_name '.slx']);

if exist(model_dir, 'dir')
    pathFolders = strsplit(path, pathsep);
    if ~any(strcmpi(pathFolders, model_dir))
        addpath(model_dir);
        fprintf('✓ 已添加模型目录到路径: %s\n', model_dir);
    end
else
    warning('模型目录不存在: %s', model_dir);
end

fis_file = fullfile(model_dir, 'Economic_Reward.fis');
if ~exist(fis_file, 'file')
    warning('未找到经济奖励FIS文件: %s', fis_file);
else
    fprintf('✓ 已检测到电价模糊规则文件: %s\n', fis_file);
end

if ~exist(model_path, 'file')
    error('模型文件不存在: %s', model_path);
end

% 加载模型
if ~bdIsLoaded(model_name)
    load_system(model_path);
    fprintf('✓ 模型已加载: %s\n', model_name);
else
    fprintf('✓ 模型已在内存中: %s\n', model_name);
end

% 设置仿真参数
simulation_days = 30;  % 训练周期30天
simulation_time = simulation_days * 24 * 3600;  % 秒
set_param(model_name, 'StopTime', num2str(simulation_time));

fprintf('✓ 仿真配置完成\n');
fprintf('  - 仿真时长: %d 天\n', simulation_days);
fprintf('  - 总秒数: %d 秒\n', simulation_time);

%% 3. 定义RL环境
% =========================================================================
fprintf('\n=== 步骤3: 定义强化学习环境 ===\n');

% 观测空间定义
numObservations = 9;  % [Battery_SOC, P_pv, P_load_in, price_profile, P_net_load, P_batt, P_grid, reward, time_of_day]
obsInfo = rlNumericSpec([numObservations 1]);
obsInfo.Name = 'Microgrid Observations';
obsInfo.Description = ['Battery_SOC, PV power, Load power, Electricity price, Net load, ', ...
    'Battery power, Grid power, reward, time of day'];

fprintf('✓ 观测空间: %d 维\n', numObservations);
fprintf('  [Battery_SOC, P_pv, P_load_in, Price, P_net_load, P_batt, P_grid, reward, time_of_day]\n');

% 动作空间定义
numActions = 1;  % 电池充放电功率
actInfo = rlNumericSpec([numActions 1]);
actInfo.Name = 'Battery Power';
actInfo.LowerLimit = -10e3;  % -10kW (放电)
actInfo.UpperLimit = 10e3;   % +10kW (充电)

fprintf('✓ 动作空间: %d 维\n', numActions);
fprintf('  电池功率范围: %.1f kW ~ %.1f kW\n', ...
    actInfo.LowerLimit/1000, actInfo.UpperLimit/1000);

% 创建Simulink环境
agentBlk = [model_name '/RL Agent'];
agentVarName = ensureAgentWorkspaceVariable(agentBlk, obsInfo, actInfo);
env = rlSimulinkEnv(model_name, agentBlk, obsInfo, actInfo);
env.ResetFcn = @(in) localResetFcn(in);

fprintf('✓ RL环境创建成功\n');
fprintf('  - Agent模块: %s\n', agentBlk);

%% 4. 训练DDPG智能体
% =========================================================================
fprintf('\n=== 步骤4: 开始DDPG训练 ===\n');

try
    % 调用训练函数
    initialAgent = [];
    if ~isempty(agentVarName)
        try
            initialAgent = evalin('base', agentVarName);
        catch
            initialAgent = [];
        end
    end
    [agent, training_results] = train_model(env, initialAgent);
    if ~isempty(agentVarName)
        assignin('base', agentVarName, agent);
        fprintf('  - 已更新工作区变量: %s (训练后的智能体)\n', agentVarName);
    end
    
    fprintf('\n✓ 训练完成!\n');
    fprintf('  - 总回合数: %d\n', training_results.total_episodes);
    fprintf('  - 训练时长: %.2f 分钟\n', training_results.training_time/60);
    
    %% 5. 保存结果
    % =====================================================================
    fprintf('\n=== 步骤5: 保存训练结果 ===\n');
    
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    save_filename = sprintf('ddpg_agent_%s.mat', timestamp);
    save(save_filename, 'agent', 'training_results');
    
    fprintf('✓ 智能体已保存: %s\n', save_filename);
    
catch ME
    fprintf('\n✗ 训练失败: %s\n', ME.message);
    fprintf('错误位置: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    rethrow(ME);
end

fprintf('\n========================================\n');
fprintf('  训练流程完成\n');
fprintf('========================================\n');

%% 辅助函数
% =========================================================================

function in = localResetFcn(in)
    % 环境重置函数
    % 初始化SOC为50%
    in = setVariable(in, 'initial_soc', 50);
end

function hours = getTimeseriesDurationHours(ts)
    % 计算timeseries对象覆盖的总小时数 (支持缺省Time向量)
    hours = 0;
    if ~isa(ts, 'timeseries')
        return;
    end

    sample_count = getTimeseriesSampleCount(ts);
    if sample_count <= 1
        return;
    end

    time_vector = ts.Time;
    if ~isempty(time_vector)
        time_vector = time_vector(:);
        if isa(time_vector, 'datetime')
            time_numeric = seconds(time_vector - time_vector(1));
        elseif isa(time_vector, 'duration')
            time_numeric = seconds(time_vector);
        else
            time_numeric = double(time_vector);
        end
        time_numeric = time_numeric(:);
        diffs = diff(time_numeric);
        diffs = diffs(diffs > 0);
        if ~isempty(diffs)
            avg_dt = median(diffs);
            hours = sample_count * avg_dt / 3600;
            return;
        end
    end

    % 若Time向量为空或无效，尝试使用 TimeInfo 推断采样时间
    increment = ts.TimeInfo.Increment;
    if isa(increment, 'duration')
        increment = seconds(increment);
    end
    if isempty(increment) || increment <= 0
        t_start = ts.TimeInfo.Start;
        t_end = ts.TimeInfo.End;
        if isa(t_start, 'duration')
            t_start = seconds(t_start);
        end
        if isa(t_end, 'duration')
            t_end = seconds(t_end);
        end
        if ~isempty(t_start) && ~isempty(t_end) && t_end > t_start
            increment = double(t_end - t_start) / (sample_count - 1);
        else
            return;
        end
    end

    hours = sample_count * double(increment) / 3600;
end

function sample_count = getTimeseriesSampleCount(ts)
    % 获取timeseries对象的样本数
    sample_count = 0;
    if ~isa(ts, 'timeseries') || isempty(ts.Data)
        return;
    end
    if ts.IsTimeFirst
        sample_count = size(ts.Data, 1);
    else
        sample_count = size(ts.Data, 2);
    end
end

function agentVarName = ensureAgentWorkspaceVariable(agentBlk, obsInfo, actInfo)
    % 确保RL Agent模块引用的变量存在并为有效的RL智能体
    agentVarName = '';
    try
        agentVarName = strtrim(get_param(agentBlk, 'Agent'));
    catch
        return;
    end

    if isempty(agentVarName) || ~isvarname(agentVarName)
        agentVarName = '';
        return;
    end

    existingAgent = [];
    try
        existingAgent = evalin('base', agentVarName);
    catch
        existingAgent = [];
    end

    if isa(existingAgent, 'rl.agent.Agent')
        return;
    end

    try
        placeholderAgent = createDefaultDDPGAgent(obsInfo, actInfo);
        assignin('base', agentVarName, placeholderAgent);
        fprintf('  - 已创建 RL Agent 占位变量: %s (DDPG结构)\n', agentVarName);
    catch ME
        warning(E.message,'无法创建RL Agent占位变量: %s', ME.message);
    end
end

function agent = createDefaultDDPGAgent(obsInfo, actInfo)
    % 创建默认的DDPG智能体 (与train_model结构保持一致)

    criticLayerSizes = [128, 64];
    statePath = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'observation')
        fullyConnectedLayer(criticLayerSizes(1), 'Name', 'CriticStateFC1')
        reluLayer('Name', 'CriticStateRelu1')
    ];
    actionPath = [
        featureInputLayer(actInfo.Dimension(1), 'Normalization', 'none', 'Name', 'action')
        fullyConnectedLayer(criticLayerSizes(1), 'Name', 'CriticActionFC1')
    ];
    commonPath = [
        additionLayer(2, 'Name', 'add')
        reluLayer('Name', 'CriticCommonRelu1')
        fullyConnectedLayer(criticLayerSizes(2), 'Name', 'CriticFC2')
        reluLayer('Name', 'CriticRelu2')
        fullyConnectedLayer(1, 'Name', 'CriticOutput')
    ];

    criticNetwork = layerGraph(statePath);
    criticNetwork = addLayers(criticNetwork, actionPath);
    criticNetwork = addLayers(criticNetwork, commonPath);
    criticNetwork = connectLayers(criticNetwork, 'CriticStateRelu1', 'add/in1');
    criticNetwork = connectLayers(criticNetwork, 'CriticActionFC1', 'add/in2');

    criticOptions = rlRepresentationOptions(...
        'LearnRate', 1e-3, ...
        'GradientThreshold', 1, ...
        'UseDevice', 'cpu');

    critic = rlQValueRepresentation(criticNetwork, obsInfo, actInfo, ...
        'Observation', {'observation'}, ...
        'Action', {'action'}, ...
        criticOptions);

    actorLayerSizes = [128, 64];
    % Actor网络设计: 输出连续的功率信号
    actorNetwork = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'observation')
        fullyConnectedLayer(actorLayerSizes(1), 'Name', 'ActorFC1')
        reluLayer('Name', 'ActorRelu1')
        fullyConnectedLayer(actorLayerSizes(2), 'Name', 'ActorFC2')
        reluLayer('Name', 'ActorRelu2')
        fullyConnectedLayer(actInfo.Dimension(1), 'Name', 'ActorFC3')
        tanhLayer('Name', 'ActorTanh')  % 输出[-1, 1]
    ];
    
    % 将tanh输出[-1,1]缩放到动作空间[-UpperLimit, +UpperLimit]
    actorNetwork = [
        actorNetwork
        scalingLayer('Name', 'ActorScaling', 'Scale', actInfo.UpperLimit)
    ];

    actorOptions = rlRepresentationOptions(...
        'LearnRate', 5e-4, ...
        'GradientThreshold', 1, ...
        'UseDevice', 'cpu');

    actor = rlDeterministicActorRepresentation(actorNetwork, obsInfo, actInfo, ...
        'Observation', {'observation'}, ...
        actorOptions);

    agentOptions = rlDDPGAgentOptions(...
        'SampleTime', 3600, ...
        'TargetSmoothFactor', 1e-3, ...
        'DiscountFactor', 0.99, ...
        'MiniBatchSize', 64, ...
        'ExperienceBufferLength', 1e6);
    noise_span = (actInfo.UpperLimit - actInfo.LowerLimit) / 2;
    base_noise = 0.2 * noise_span;
    agentOptions.NoiseOptions.Mean = 0;
    agentOptions.NoiseOptions.MeanAttractionConstant = 1e-4;
    agentOptions.NoiseOptions.Variance = base_noise^2;
    agentOptions.NoiseOptions.VarianceDecayRate = 5e-5;

    agent = rlDDPGAgent(actor, critic, agentOptions);
end
