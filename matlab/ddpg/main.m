%% 微电网DDPG强化学习 - 统一主程序
% =========================================================================
% 功能: 训练DDPG智能体优化微电网储能调度
% 特色: 合并main.m和main_simplified.m的所有优点
% 算法: Deep Deterministic Policy Gradient (DDPG)
% 版本: 统一增强版 v3.0
% =========================================================================

function main(training_mode)
    % 输入参数:
    %   training_mode - 可选，训练模式选择
    %     'auto'     - 自动选择最佳训练方法 (默认)
    %     'advanced' - 使用高级训练方法  
    %     'simple'   - 使用兼容训练方法
    %     'legacy'   - 使用原始训练方法
    
    if nargin < 1
        training_mode = 'auto';
    end
    
    clc;
    close all;
    
    fprintf('========================================\n');
    fprintf('  微电网DDPG强化学习训练系统 v3.0\n');
    fprintf('========================================\n');
    start_time = datetime("now", "Format", "yyyy-MM-dd HH:mm:ss");
    fprintf('启动时间: %s\n', char(start_time));
    fprintf('训练模式: %s\n', upper(training_mode));
    fprintf('========================================\n\n');
    
    try
        %% 1. 环境初始化
        fprintf('=== 步骤1: 环境初始化 ===\n');
        config = initialize_environment();
        fprintf('✓ 环境初始化完成\n');
        
        %% 1.5. 加载模糊逻辑系统
        try
            load_fuzzy_logic_files();
            fprintf('✓ 模糊逻辑系统加载完成\n');
        catch ME
            fprintf('⚠ 模糊逻辑系统加载失败: %s\n', ME.message);
            fprintf('  程序将继续运行，但可能影响Simulink模型\n');
        end
        
        %% 2. 数据加载与验证
        fprintf('\n=== 步骤2: 数据加载与验证 ===\n');
        data = load_and_validate_data(config);
        fprintf('✓ 数据加载完成 (%.0f小时)\n', data.duration_hours);
        
        %% 3. Simulink环境配置
        fprintf('\n=== 步骤3: Simulink环境配置 ===\n');
        [env, agent_var_name] = setup_simulink_environment(config, data);
        fprintf('✓ Simulink环境配置完成\n');
        
        %% 4. 智能体训练
        fprintf('\n=== 步骤4: 开始DDPG训练 ===\n');
        fprintf('训练模式: %s\n', training_mode);
        
        % 获取初始智能体（如果存在）
        initial_agent = [];
        if ~isempty(agent_var_name)
            try
                initial_agent = evalin('base', agent_var_name);
            catch
                initial_agent = [];
            end
        end
        
        tic;
        [agent, results] = execute_training(env, initial_agent, training_mode, config);
        training_time = toc;
        
        fprintf('✓ 训练完成 (%.1f分钟)\n', training_time/60);
        
        %% 5. 保存结果
        fprintf('\n=== 步骤5: 保存训练结果 ===\n');
        save_results(agent, results, config, training_mode, agent_var_name);
        fprintf('✓ 结果保存完成\n');
        
        %% 6. 最终验证
        fprintf('\n=== 步骤6: 最终验证 ===\n');
        verify_final_performance(agent, env);
        
    catch ME
        fprintf('\n✗ 错误: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('位置: %s (第%d行)\n', ME.stack(1).name, ME.stack(1).line);
        end
        rethrow(ME);
    end
    
    fprintf('\n========================================\n');
    fprintf('  训练流程全部完成！\n');
    fprintf('========================================\n');
end

%% 1. 环境初始化
function config = initialize_environment()
    % 路径配置
    project_root = fileparts(fileparts(pwd));
    model_dir = fullfile(project_root, 'model');
    
    % 添加关键路径
    addpath(genpath(project_root));
    addpath(model_dir);
    
    % 配置参数
    config.model_name = 'Microgrid';
    config.model_path = fullfile(model_dir, [config.model_name '.slx']);
    config.data_path = fullfile(project_root, 'matlab', 'src', 'microgrid_simulation_data.mat');
    config.simulation_days = 30;
    config.simulation_time = config.simulation_days * 24 * 3600;
    config.model_dir = model_dir;
    
    % 文件存在性检查
    if ~exist(config.model_path, 'file')
        error('Simulink模型不存在: %s', config.model_path);
    end
    if ~exist(config.data_path, 'file')
        error('数据文件不存在: %s\n请先运行 matlab/src/generate_data.m', config.data_path);
    end
end

%% 2. 数据加载与验证（增强版）
function data = load_and_validate_data(config)
    fprintf('正在加载仿真数据...\n');
    raw_data = load(config.data_path);
    
    % 数据兼容性处理
    required_vars = {'pv_power_profile', 'load_power_profile'};
    for i = 1:length(required_vars)
        var_name = required_vars{i};
        if ~isfield(raw_data, var_name)
            error('缺少必要数据: %s', var_name);
        end
    end
    
    % 电价数据处理（兼容多种格式）
    if isfield(raw_data, 'price_profile')
        data.price_profile = raw_data.price_profile;
    elseif isfield(raw_data, 'price_data')
        data.price_profile = raw_data.price_data;
        fprintf('  - 已转换price_data为price_profile\n');
    else
        error('缺少电价数据: price_profile 或 price_data');
    end
    
    % timeseries格式检查和修复
    if ~isa(data.price_profile, 'timeseries')
        if isnumeric(data.price_profile)
            % 转换为timeseries
            if isfield(raw_data, 'time_seconds')
                time_vector = raw_data.time_seconds(:);
            else
                time_vector = (0:length(data.price_profile)-1)' * 3600;
            end
            data.price_profile = timeseries(data.price_profile(:), time_vector);
            data.price_profile.Name = 'Electricity Price Profile';
            data.price_profile.DataInfo.Units = 'CNY/kWh';
            fprintf('  - 已重建电价时间序列\n');
        else
            error('无法处理电价数据格式: %s', class(data.price_profile));
        end
    end
    
    % 数据复制
    data.pv_power = raw_data.pv_power_profile;
    data.load_power = raw_data.load_power_profile;
    
    % 复制其他字段
    field_names = fieldnames(raw_data);
    for i = 1:length(field_names)
        field_name = field_names{i};
        if ~isfield(data, field_name)
            data.(field_name) = raw_data.(field_name);
        end
    end
    
    % 验证数据长度
    data.duration_hours = calculate_duration_hours(data.pv_power);
    if data.duration_hours < 700
        error('数据长度不足: 需要至少720小时，当前仅有%.0f小时', data.duration_hours);
    end
    
    fprintf('  - 光伏数据: %.0f 小时\n', data.duration_hours);
    fprintf('  - 负载数据: %.0f 小时\n', calculate_duration_hours(data.load_power));
    fprintf('  - 电价数据: %.0f 小时\n', calculate_duration_hours(data.price_profile));

    % 将关键数据推送到基础工作区，供Simulink模型使用
    push_simulation_data_to_base(data, raw_data);
end

%% 3. Simulink环境设置（增强版）
function [env, agent_var_name] = setup_simulink_environment(config, ~)
    % 加载模型
    if ~bdIsLoaded(config.model_name)
        load_system(config.model_path);
    end
    
    % 设置仿真参数
    set_param(config.model_name, 'StopTime', num2str(config.simulation_time));
    
    % 定义观测和动作空间
    obs_info = rlNumericSpec([9 1]);
    obs_info.Name = 'Microgrid Observations';
    obs_info.Description = 'Battery_SOC, P_pv, P_load, Price, P_net_load, P_batt, P_grid, reward, time_of_day';
    
    act_info = rlNumericSpec([1 1]);
    act_info.Name = 'Battery Power';
    act_info.LowerLimit = -10e3;
    act_info.UpperLimit = 10e3;
    act_info.Description = 'Continuous battery charge/discharge power';
    
    % 确保agentObj变量存在
    agent_block = [config.model_name '/RL Agent'];
    agent_var_name = ensure_agent_workspace_variable(agent_block, obs_info, act_info);
    
    % 创建Simulink环境
    env = rlSimulinkEnv(config.model_name, agent_block, obs_info, act_info);
    env.ResetFcn = @(in) setVariable(in, 'initial_soc', 50);
    
    fprintf('  - 观测维度: %d\n', obs_info.Dimension(1));
    fprintf('  - 动作范围: [%.1f, %.1f] kW\n', act_info.LowerLimit/1000, act_info.UpperLimit/1000);
    fprintf('  - agentObj变量: %s\n', agent_var_name);
end

%% 4. 执行训练（统一接口）
function [agent, results] = execute_training(env, initial_agent, training_mode, ~)
    switch lower(training_mode)
        case 'auto'
            % 自动选择：优先高级 -> 兼容 -> 原始
            fprintf('使用自动模式，尝试最佳训练方法...\n');
            try
                fprintf('  -> 尝试高级训练方法\n');
                [agent, results] = train_model(env, initial_agent, 'advanced');
                fprintf('✓ 高级训练方法成功\n');
            catch ME1
                fprintf('  -> 高级方法失败: %s\n', ME1.message);
                try
                    fprintf('  -> 尝试兼容训练方法\n');
                    [agent, results] = train_model(env, initial_agent, 'simple');
                    fprintf('✓ 兼容训练方法成功\n');
                catch ME2
                    fprintf('  -> 兼容方法失败: %s\n', ME2.message);
                    fprintf('  -> 使用原始训练方法\n');
                    [agent, results] = train_model(env, initial_agent, 'legacy');
                    fprintf('✓ 原始训练方法成功\n');
                end
            end
            
        case {'advanced', 'simple', 'legacy'}
            [agent, results] = train_model(env, initial_agent, training_mode);
            
        otherwise
            error('未知训练模式: %s', training_mode);
    end
end

%% 5. 保存结果（增强版）
function save_results(agent, results, config, training_mode, agent_var_name)
    % 更新工作区变量
    if ~isempty(agent_var_name)
        assignin('base', agent_var_name, agent);
        fprintf('  - 已更新工作区变量: %s\n', agent_var_name);
    end
    
    % 保存到文件
    timestamp_dt = datetime("now", "Format", "yyyyMMdd_HHmmss");
    timestamp = char(timestamp_dt);
    filename = sprintf('ddpg_agent_%s_%s.mat', training_mode, timestamp);
    
    % 保存额外信息
    save_info = struct();
    save_info.training_mode = training_mode;
    save_info.matlab_version = version;
    save_info.timestamp = timestamp_dt;
    save_info.config = config;
    
    save(filename, 'agent', 'results', 'save_info');
    fprintf('  - 文件: %s\n', filename);
    
    % 显示结果统计
    if isfield(results, 'total_episodes')
        fprintf('  - 训练回合: %d\n', results.total_episodes);
        fprintf('  - 训练时长: %.2f分钟\n', results.training_time/60);
        if isfield(results, 'best_reward')
            fprintf('  - 最佳奖励: %.1f\n', results.best_reward);
        end
    end
end

%% 6. 最终性能验证
function verify_final_performance(agent, env)
    try
        obs_info = getObservationInfo(env);
        act_info = getActionInfo(env);
        
        % 测试连续动作生成
        fprintf('测试连续动作生成...\n');
        agent.UseExplorationPolicy = false;
        
        num_samples = 20;
        actions = zeros(1, num_samples);
        for i = 1:num_samples
            test_obs = rand(obs_info.Dimension) * 0.5 + 0.25;
            action_cell = getAction(agent, {test_obs});
            actions(i) = action_cell{1};
        end
        
        fprintf('  - 动作范围: [%.2f, %.2f] kW\n', min(actions)/1000, max(actions)/1000);
        fprintf('  - 动作方差: %.2f kW²\n', var(actions)/1000);
        fprintf('  - 唯一值: %d / %d\n', length(unique(round(actions, 1))), num_samples);
        
        % 验证连续性
        is_continuous = length(unique(round(actions, 1))) > num_samples * 0.7;
        in_range = all(actions >= act_info.LowerLimit-100) && all(actions <= act_info.UpperLimit+100);
        has_variation = std(actions) > 50;
        
        if is_continuous && in_range && has_variation
            fprintf('✅ 连续动作输出验证通过！\n');
        else
            fprintf('⚠️ 连续动作输出可能需要优化\n');
        end
        
    catch ME
        fprintf('最终验证失败: %s\n', ME.message);
    end
end

%% 辅助函数

% 模糊逻辑文件加载
function load_fuzzy_logic_files()
    project_root = fileparts(fileparts(pwd));
    model_dir = fullfile(project_root, 'model');
    
    if exist(model_dir, 'dir')
        addpath(model_dir);
    end
    
    fis_files = {
        'Battery_State_Type2.fis'
        'Battery_State.fis'
        'Economic_Reward.fis'
        'Economic_Decision.fis'
    };
    
    for i = 1:length(fis_files)
        fis_path = fullfile(model_dir, fis_files{i});
        if exist(fis_path, 'file')
            try
                fis = readfis(fis_path);
                var_name = strrep(fis_files{i}, '.fis', '');
                assignin('base', var_name, fis);
            catch
                % 忽略个别文件加载错误
            end
        end
    end
    
    % 创建必要变量
    if ~evalin('base', 'exist(''eml_transient'', ''var'')')
        assignin('base', 'eml_transient', struct());
    end
    
    fuzzy_vars = {'price_norm', 0.5; 'soc_current', 50; 'soc_diff', 0; 'reward_signal', 0};
    for i = 1:size(fuzzy_vars, 1)
        var_name = fuzzy_vars{i, 1};
        var_value = fuzzy_vars{i, 2};
        if ~evalin('base', sprintf('exist(''%s'', ''var'')', var_name))
            assignin('base', var_name, var_value);
        end
    end
end

% 将仿真数据分发到基础工作区
function push_simulation_data_to_base(data, raw_data)
    % 优先使用经过校验的数据对象
    base_assignments = {
        'price_profile', data.price_profile;
        'pv_power_profile', data.pv_power;
        'load_power_profile', data.load_power
    };
    for i = 1:size(base_assignments, 1)
        var_name = base_assignments{i, 1};
        var_value = base_assignments{i, 2};
        assignin('base', var_name, var_value);
    end

    % 其余原始数据字段也同步到基础工作区，避免模型依赖缺失
    raw_fields = fieldnames(raw_data);
    for i = 1:numel(raw_fields)
        field = raw_fields{i};
        if ismember(field, {'price_profile', 'pv_power_profile', 'load_power_profile'})
            continue; % 已使用处理后的版本
        end
        assignin('base', field, raw_data.(field));
    end
end

% 确保智能体变量存在
function agent_var_name = ensure_agent_workspace_variable(agent_block, obs_info, act_info)
    agent_var_name = '';
    try
        agent_var_name = strtrim(get_param(agent_block, 'Agent'));
    catch
        return;
    end
    
    if isempty(agent_var_name) || ~isvarname(agent_var_name)
        agent_var_name = '';
        return;
    end
    
    % 检查现有变量
    try
        existing_agent = evalin('base', agent_var_name);
    catch
        existing_agent = [];
    end
    
    if isa(existing_agent, 'rl.agent.Agent')
        return;
    end
    
    % 创建占位符智能体
    try
        placeholder_agent = create_placeholder_agent(obs_info, act_info);
        assignin('base', agent_var_name, placeholder_agent);
        fprintf('  - 已创建占位符智能体: %s\n', agent_var_name);
    catch ME
        fprintf('  - 占位符智能体创建失败: %s\n', ME.message);
    end
end

% 创建占位符智能体
function placeholder_agent = create_placeholder_agent(obs_info, act_info)
    action_range = act_info.UpperLimit - act_info.LowerLimit;
    action_scale = action_range / 2;
    action_bias = (act_info.UpperLimit + act_info.LowerLimit) / 2;
    
    % 简化Actor网络
    actor_layers = [
        featureInputLayer(obs_info.Dimension(1), 'Name', 'state')
        fullyConnectedLayer(32, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(act_info.Dimension(1), 'Name', 'fc2')
        tanhLayer('Name', 'tanh')
        scalingLayer('Name', 'scaling', 'Scale', action_scale, 'Bias', action_bias)
    ];
    
    actor_options = rlRepresentationOptions('LearnRate', 1e-3);
    actor = rlDeterministicActorRepresentation(actor_layers, obs_info, act_info, ...
        'Observation', {'state'}, actor_options);
    
    % 简化Critic网络
    critic_layers = [
        featureInputLayer(obs_info.Dimension(1) + act_info.Dimension(1), 'Name', 'input')
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(1, 'Name', 'output')
    ];
    
    critic_options = rlRepresentationOptions('LearnRate', 1e-3);
    critic = rlQValueRepresentation(critic_layers, obs_info, act_info, ...
        'Observation', {'input'}, critic_options);
    
    agent_options = rlDDPGAgentOptions();
    agent_options.SampleTime = 3600;
    agent_options.DiscountFactor = 0.99;
    
    placeholder_agent = rlDDPGAgent(actor, critic, agent_options);
end

% 计算时长
function hours = calculate_duration_hours(ts)
    hours = 0;
    if ~isa(ts, 'timeseries')
        return;
    end
    
    if isempty(ts.Time)
        sample_count = length(ts.Data);
        hours = sample_count;
    else
        time_span = ts.Time(end) - ts.Time(1);
        if isa(time_span, 'duration')
            hours = hours(time_span);
        else
            hours = time_span / 3600;
        end
    end
end
