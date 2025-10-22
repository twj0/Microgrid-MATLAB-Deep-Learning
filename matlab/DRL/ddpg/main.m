function main(options)
if nargin < 1 || isempty(options)
    options = struct();
end

clc;
close all;

fprintf('========================================\n');
fprintf('  Microgrid DDPG Reinforcement Learning Training v1.0\n');
fprintf('========================================\n');
fprintf('Start time: %s\n', char(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss")));

try
    config = initialize_environment(options);
    fprintf('✓ Environment paths ready\n');

    try
        load_fuzzy_logic_files(config.model_dir);
        fprintf('✓ Fuzzy logic system loaded\n');
    catch ME
        fprintf('⚠ Failed to load fuzzy logic system: %s\n', ME.message);
    end

    [data, raw_data] = load_and_validate_data(config);
    fprintf('  - PV data: %.0f hours\n', data.duration_hours);
    fprintf('  - Load data: %.0f hours (%d samples)\n', calculate_duration_hours(data.load_power), calculate_sample_count(data.load_power));
    fprintf('  - Price data: %.0f hours (%d samples)\n', calculate_duration_hours(data.price_profile), calculate_sample_count(data.price_profile));
    push_simulation_data_to_base(data, raw_data);
    fprintf('✓ Data loaded and assigned\n');

    [env, agent_var_name] = setup_simulink_environment(config);
    fprintf('✓ Simulink environment configured\n');

    try
        exists_flag = evalin('base','exist(''g_episode_num'',''var'')');
    catch
        exists_flag = 0;
    end
    if ~exists_flag
        g_episode_num = Simulink.Parameter(0);
        g_episode_num.CoderInfo.StorageClass = 'ExportedGlobal';
        assignin('base','g_episode_num', g_episode_num);
    else
        try
            g = evalin('base','g_episode_num');
            if isa(g,'Simulink.Parameter')
                g.Value = 0;
                assignin('base','g_episode_num', g);
            else
                assignin('base','g_episode_num', 0);
            end
        catch
            assignin('base','g_episode_num', 0);
        end
    end

    env.ResetFcn = @reset_episode_counter;

    initial_agent = [];
    if ~isempty(agent_var_name)
        try
            initial_agent = evalin('base', agent_var_name);
        catch
            initial_agent = [];
        end
    end

    training_options = build_training_config(options);
    [agent, results] = train_model(env, initial_agent, training_options);
    fprintf('✓ DDPG training completed\n');

    if exist('results', 'var') && ~isempty(results)
        assignin('base', 'ddpg_training_results', results);
    end

    save_results(agent, results, config, agent_var_name);
    fprintf('✓ Training artifacts saved\n');

    verify_final_performance(agent, env);
    fprintf('✓ Policy verification finished\n');

    % 可视化输出（与SAC/PPO/DQN保持一致）
    try
        run_visualization(config, results);
    catch vizErr
        fprintf('⚠ Visualization failed: %s\n', vizErr.message);
    end


catch ME
    fprintf('\n✗ Error: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('Location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    rethrow(ME);
end

fprintf('========================================\n');
fprintf('  Training pipeline finished\n');
fprintf('========================================\n');
end

%% 1. 环境初始化
function config = initialize_environment(options)
if nargin < 1
    options = struct();
end
    % 路径配置
    script_dir = fileparts(mfilename('fullpath'));
    project_root = find_project_root(script_dir);
    model_dir = fullfile(project_root, 'model');

    % 添加关键路径
    addpath(genpath(project_root));
    addpath(model_dir);

    % 配置参数
    config.model_name = 'Microgrid';
    config.model_path = fullfile(model_dir, [config.model_name '.slx']);
    config.data_path = fullfile(project_root, 'matlab', 'src', 'microgrid_simulation_data.mat');
    config.simulation_days = get_option(options, 'simulation_days', 30);
    config.sample_time = 3600;
    config.simulation_time = config.simulation_days * 24 * config.sample_time;
    config.model_dir = model_dir;

    % 文件存在性检查
    if ~exist(config.model_path, 'file')
        error('Simulink模型不存在: %s', config.model_path);
    end
    if ~exist(config.data_path, 'file')
        error('数据文件不存在: %s\n请先运行 matlab/src/generate_data.m', config.data_path);
    end
end

function in = reset_episode_counter(in)
persistent ep
if isempty(ep)
    ep = 0;
end
ep = ep + 1;
assignin('base','g_episode_num', ep);
in = setVariable(in, 'g_episode_num', ep);
in = setVariable(in, 'initial_soc', 50);
end

%% 2. 数据加载与验证（增强版）
function [data, raw_data] = load_and_validate_data(config)
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
    data.sample_count = calculate_sample_count(data.pv_power);
    required_samples = ceil(config.simulation_time / config.sample_time);

    if data.sample_count < required_samples
        deficit = required_samples - data.sample_count;
        max_auto_pad = 24;
        if deficit <= max_auto_pad
            fprintf(['⚠ 数据样本不足: 缺少%d个样本（约%.0f小时），', ...
                     '自动复制最后一个时刻的数据进行补齐。\n'], ...
                deficit, deficit * (config.sample_time/3600));

            data.price_profile = pad_timeseries_to_length(data.price_profile, required_samples, config.sample_time);
            data.pv_power = pad_timeseries_to_length(data.pv_power, required_samples, config.sample_time);
            data.load_power = pad_timeseries_to_length(data.load_power, required_samples, config.sample_time);

            if isfield(raw_data, 'pv_power_profile')
                raw_data.pv_power_profile = data.pv_power;
            end
            if isfield(raw_data, 'load_power_profile')
                raw_data.load_power_profile = data.load_power;
            end
            if isfield(raw_data, 'price_profile')
                raw_data.price_profile = data.price_profile;
            end
            if isfield(raw_data, 'price_data')
                raw_data.price_data = pad_timeseries_to_length(raw_data.price_data, required_samples, config.sample_time);
            end

            data.sample_count = calculate_sample_count(data.pv_power);
        else
            required_hours = required_samples * (config.sample_time/3600);
            error(['数据样本不足: 需要>=%d个样本（约%.0f小时）', ...
                   '，当前仅有%d个样本（约%.0f小时）。\n', ...
                   '请运行 matlab/src/generate_data.m 重新生成足够长度的时序数据。'], ...
                required_samples, required_hours, ...
                data.sample_count, data.sample_count * (config.sample_time/3600));
        end
    end

    data.duration_hours = calculate_duration_hours(data.pv_power);
    required_hours = config.simulation_time / 3600;
    tolerance_hours = max(config.sample_time / 3600, 1e-6);
    if data.duration_hours + tolerance_hours < required_hours
        error('数据长度不足: 需要至少%.0f小时，当前仅有%.0f小时', required_hours, data.duration_hours);
    end

    fprintf('  - 光伏数据: %.0f 小时\n', data.duration_hours);
    fprintf('  - 负载数据: %.0f 小时\n', calculate_duration_hours(data.load_power));
    fprintf('  - 电价数据: %.0f 小时\n', calculate_duration_hours(data.price_profile));

    % 将关键数据推送到基础工作区，供Simulink模型使用
    push_simulation_data_to_base(data, raw_data);
end

%% 3. Simulink环境设置（增强版）
function [env, agent_var_name] = setup_simulink_environment(config)
    % 加载模型
    if ~bdIsLoaded(config.model_name)
        load_system(config.model_path);
    end

    % 设置仿真参数
    set_param(config.model_name, 'StopTime', num2str(config.simulation_time));

    % 定义观测和动作空间
    obs_info = rlNumericSpec([12 1]);
    obs_info.Name = 'Microgrid Observations';
    obs_info.Description = 'Battery_SOH, Battery_SOC, P_pv, P_load, Price, P_net_load, P_batt, P_grid, fuzzy_reward, economic_reward, health_reward, time_of_day';

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
    env.ResetFcn = @reset_episode_counter;

    fprintf('  - 观测维度: %d\n', obs_info.Dimension(1));
    fprintf('  - 奖励系统: 自洽跟踪 + 经济 + 健康 + 模糊修正\n');
    fprintf('  - 动作范围: [%.1f, %.1f] kW\n', act_info.LowerLimit/1000, act_info.UpperLimit/1000);
    fprintf('  - agentObj变量: %s\n', agent_var_name);
end

%% 4. 执行训练（统一接口）
function [agent, results] = execute_training(env, initial_agent, training_mode, ~)
    training_options = build_training_config(struct());
    training_options.MaxEpisodes = 1000;
    training_options.MaxStepsPerEpisode = 24*3600/config.sample_time;
    training_options.ScoreAveragingWindowLength = 100;
    training_options.SaveAgentCriteria = 'EpisodeReward';
    training_options.SaveAgentValue = Inf;
    training_options.StopTrainingCriteria = 'AverageReward';
    training_options.StopTrainingValue = 1000;
    training_options.StopOnError = 'on';

    [agent, results] = train(env, initial_agent, training_options);
end

%% 5. 保存结果（增强版）
function save_results(agent, results, config, agent_var_name)
    if ~isempty(agent_var_name)
        assignin('base', agent_var_name, agent);
        fprintf('  - Updated workspace variable: %s\n', agent_var_name);
    end

    timestamp_dt = datetime("now", "Format", "yyyyMMdd_HHmmss");
    timestamp = char(timestamp_dt);
    filename = sprintf('ddpg_agent_%s.mat', timestamp);

    save_info = struct();
    save_info.matlab_version = version;
    save_info.timestamp = timestamp_dt;
    save_info.config = config;

    save(filename, 'agent', 'results', 'save_info');
    fprintf('  - File saved: %s\n', filename);

    if isfield(results, 'total_episodes')
        fprintf('  - Episodes: %d\n', results.total_episodes);
        fprintf('  - Best reward: %.1f\n', results.best_reward);
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
function load_fuzzy_logic_files(model_dir)
    if nargin < 1 || isempty(model_dir)
        script_dir = fileparts(mfilename('fullpath'));
        project_root = find_project_root(script_dir);
        model_dir_default = fullfile(project_root, 'model');
        model_dir = model_dir_default;
    end

    if exist(model_dir, 'dir')
        addpath(model_dir);
    end

    fis_files = {
        'Battery_State_Type2.fis'
        'Battery_State.fis'
        'Economic_Reward.fis'
        'Economic_Decision.fis'
        'fuzzy_correction.fis'
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
        featureInputLayer(obs_info.Dimension(1), 'Name', 'state', 'Normalization', 'none')
        fullyConnectedLayer(32, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(act_info.Dimension(1), 'Name', 'fc2')
        tanhLayer('Name', 'tanh')
        scalingLayer('Name', 'scaling', 'Scale', action_scale, 'Bias', action_bias)
    ];

    actor_options = rlRepresentationOptions('LearnRate', 1e-3);
    actor = rlDeterministicActorRepresentation(actor_layers, obs_info, act_info, ...
        'Observation', {'state'}, actor_options);

    % 简化Critic网络，使用状态和动作分支
    critic_lgraph = layerGraph();

    state_path = [
        featureInputLayer(obs_info.Dimension(1), 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(32, 'Name', 'state_fc1')
        reluLayer('Name', 'state_relu1')
    ];

    action_path = [
        featureInputLayer(act_info.Dimension(1), 'Normalization', 'none', 'Name', 'action')
        fullyConnectedLayer(32, 'Name', 'action_fc1')
        reluLayer('Name', 'action_relu1')
    ];

    common_path = [
        concatenationLayer(1, 2, 'Name', 'concat')
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(1, 'Name', 'output')
    ];

    critic_lgraph = addLayers(critic_lgraph, state_path);
    critic_lgraph = addLayers(critic_lgraph, action_path);
    critic_lgraph = addLayers(critic_lgraph, common_path);

    critic_lgraph = connectLayers(critic_lgraph, 'state_relu1', 'concat/in1');
    critic_lgraph = connectLayers(critic_lgraph, 'action_relu1', 'concat/in2');

    critic_options = rlRepresentationOptions('LearnRate', 1e-3);
    critic = rlQValueRepresentation(critic_lgraph, obs_info, act_info, ...
        'Observation', {'state'}, 'Action', {'action'}, critic_options);

    agent_options = rlDDPGAgentOptions();
    agent_options.SampleTime = 3600;
    agent_options.DiscountFactor = 0.99;

    placeholder_agent = rlDDPGAgent(actor, critic, agent_options);
end

% 计算时长
function duration = calculate_duration_hours(ts)
    duration = 0;
    if isa(ts, 'timeseries')
        if isempty(ts.Data)
            return;
        end

        if ts.IsTimeFirst
            sample_count = size(ts.Data, 1);
        else
            sample_count = size(ts.Data, 2);
        end
        if sample_count <= 1
            return;
        end

        if ~isempty(ts.Time)
            t = ts.Time;
            if isa(t, 'duration')
                numeric_time = seconds(t);
            elseif isa(t, 'datetime')
                numeric_time = seconds(t - t(1));
            else
                numeric_time = double(t);
            end
            duration = (numeric_time(end) - numeric_time(1)) / 3600;
            if duration <= 0
                duration = (sample_count - 1) * infer_time_step(ts) / 3600;
            end
        else
            duration = (sample_count - 1) * infer_time_step(ts) / 3600;
        end
        return;
    end

    if isnumeric(ts) || islogical(ts)
        sample_count = numel(ts);
        if sample_count <= 1
            duration = 0;
        else
            duration = sample_count - 1;
        end
    elseif iscell(ts)
        sample_count = numel(ts);
        if sample_count <= 1
            duration = 0;
        else
            duration = sample_count - 1;
        end
    end
end

function count = calculate_sample_count(ts)
    count = 0;
    if isa(ts, 'timeseries')
        if isempty(ts.Data)
            return;
        end
        if ts.IsTimeFirst
            count = size(ts.Data, 1);
        else
            count = size(ts.Data, 2);
        end
        return;
    end

    if isnumeric(ts) || islogical(ts)
        count = numel(ts);
    elseif iscell(ts)
        count = numel(ts);
    end
end

function dt = infer_time_step(ts)
    dt = 3600;
    increment = ts.TimeInfo.Increment;
    if isa(increment, 'duration')
        increment = seconds(increment);
    end
    if ~isempty(increment) && increment > 0
        dt = double(increment);
        return;
    end
    if ~isempty(ts.TimeInfo.End) && ~isempty(ts.TimeInfo.Start)
        t_start = ts.TimeInfo.Start;
        t_end = ts.TimeInfo.End;
        if isa(t_start, 'duration')
            t_start = seconds(t_start);
        end
        if isa(t_end, 'duration')
            t_end = seconds(t_end);
        end
        dt = double(t_end - t_start) / max(1, (ts.Length - 1));
    end
end

function ts_out = pad_timeseries_to_length(ts_in, target_samples, sample_time)
    ts_out = ts_in;

    if isa(ts_in, 'timeseries')
        ts_out = ts_in.copy;
        current_samples = calculate_sample_count(ts_out);
        missing = target_samples - current_samples;
        if missing <= 0
            return;
        end

        dt = infer_time_step(ts_out);
        if ~isfinite(dt) || dt <= 0
            dt = sample_time;
        end

        if isempty(ts_out.Time)
            base_time = (0:current_samples-1)' * dt;
            ts_out.Time = base_time;
        end

        data = ts_out.Data;
        rep_shape = ones(1, ndims(data));

        if ts_out.IsTimeFirst
            rep_shape(1) = missing;
            last_sample = data(end,:,:,:,:,:);
            pad_data = repmat(last_sample, rep_shape);
            ts_out.Data = cat(1, data, pad_data);
            if ~isempty(ts_out.Quality)
                quality = ts_out.Quality;
                last_quality = quality(end,:,:,:,:,:);
                pad_quality = repmat(last_quality, rep_shape);
                ts_out.Quality = cat(1, quality, pad_quality);
            end
        else
            rep_shape(2) = missing;
            last_sample = data(:,end,:,:,:,:);
            pad_data = repmat(last_sample, rep_shape);
            ts_out.Data = cat(2, data, pad_data);
            if ~isempty(ts_out.Quality)
                quality = ts_out.Quality;
                last_quality = quality(:,end,:,:,:,:);
                pad_quality = repmat(last_quality, rep_shape);
                ts_out.Quality = cat(2, quality, pad_quality);
            end
        end

        time_tail = build_time_extension(ts_out.Time, dt, missing);
        ts_out.Time = concatenate_time(ts_out.Time, time_tail);
        if isempty(ts_out.TimeInfo.Increment)
            if isa(ts_out.Time(end), 'duration') || isa(ts_out.Time(end), 'datetime')
                ts_out.TimeInfo.Increment = seconds(dt);
            else
                ts_out.TimeInfo.Increment = dt;
            end
        end
        return;
    end

    if isnumeric(ts_in) || islogical(ts_in)
        data = ts_in;
        if isvector(data)
            column = data(:);
            missing = target_samples - numel(column);
            if missing <= 0
                if isrow(data)
                    ts_out = data;
                else
                    ts_out = column;
                end
                return;
            end
            pad_values = repmat(column(end), missing, 1);
            padded = [column; pad_values];
            if isrow(ts_in)
                ts_out = padded.';
            else
                ts_out = padded;
            end
            return;
        end

        current_samples = size(data, 1);
        missing = target_samples - current_samples;
        if missing <= 0
            ts_out = data;
            return;
        end
        rep_shape = ones(1, ndims(data));
        rep_shape(1) = missing;
        last_sample = data(end,:,:,:,:,:);
        pad_data = repmat(last_sample, rep_shape);
        ts_out = cat(1, data, pad_data);
    elseif iscell(ts_in)
        data = ts_in(:);
        current_samples = numel(data);
        missing = target_samples - current_samples;
        if missing <= 0
            ts_out = data;
            return;
        end
        pad_values = repmat(data(end), missing, 1);
        ts_out = [data; pad_values];
    end
end

function extension = build_time_extension(time_vector, dt, missing)
    if missing <= 0
        extension = [];
        return;
    end

    if isempty(time_vector)
        extension = (0:missing-1)' * dt;
        return;
    end

    last_time = time_vector(end);
    if isa(last_time, 'duration')
        extension = last_time + seconds((1:missing)' * dt);
    elseif isa(last_time, 'datetime')
        extension = last_time + seconds((1:missing)' * dt);
    else
        extension = last_time + (1:missing)' * dt;
    end
end

function combined = concatenate_time(original, extension)
    if isempty(extension)
        combined = original;
        return;
    end
    if isempty(original)
        combined = extension;
        return;
    end

    if isrow(original)
        if iscolumn(extension)
            extension = extension.';
        end
        combined = [original, extension];
    else
        if isrow(extension)
            extension = extension.';
        end
        combined = [original; extension];
    end
end

function project_root = find_project_root(start_dir)
    project_root = start_dir;
    max_depth = 10;
    for i = 1:max_depth
        if exist(fullfile(project_root, 'matlab'), 'dir') && exist(fullfile(project_root, 'model'), 'dir')
            return;
        end
        parent_dir = fileparts(project_root);
        if isempty(parent_dir) || strcmp(parent_dir, project_root)
            break;
        end
        project_root = parent_dir;
    end
    error('DDPG:ProjectRootNotFound', '无法从路径%s定位项目根目录', start_dir);
end

function value = get_option(options, field, default_value)
if isstruct(options) && isfield(options, field)
    value = options.(field);
else
    value = default_value;
end
end

function training_options = build_training_config(options)
if nargin < 1 || isempty(options)
    options = struct();
end

training_options = options;
training_options.maxEpisodes = get_option(options, 'maxEpisodes', 2000);
training_options.maxSteps = get_option(options, 'maxSteps', 720);
training_options.stopValue = get_option(options, 'stopValue', Inf);
training_options.agentVariant = get_option(options, 'agentVariant', 'advanced');
training_options.saveBestDuringTraining = get_option(options, 'saveBestDuringTraining', false);

% 兼容旧字段
if isfield(options, 'SampleTime') && ~isfield(training_options, 'sampleTime')
    training_options.sampleTime = options.SampleTime;
end
end



%% 7. 训练后可视化（与SAC/PPO一致风格）
function run_visualization(config, trainingResults)
    if nargin < 2
        trainingResults = [];
    end
    project_root = fileparts(config.model_dir);
    matlab_src_dir = fullfile(project_root, 'matlab', 'src');
    if exist(matlab_src_dir, 'dir'), addpath(matlab_src_dir); end
    visualization_path = which('visualization');
    expected_visualization = fullfile(matlab_src_dir, 'visualization.m');
    if isempty(visualization_path) || ~strcmpi(visualization_path, expected_visualization)
        fprintf('⚠ visualization.m not found, skipping\n');
        return;
    end
    results_root = fullfile(project_root, 'results');
    if ~exist(results_root, 'dir'), mkdir(results_root); end
    timestamp = char(datetime("now", "Format", "yyyyMMdd_HHmmss"));
    ddpg_folder = fullfile(results_root, ['DDPG_' timestamp]);
    if ~exist(ddpg_folder, 'dir'), mkdir(ddpg_folder); end
    viz_options = struct('workspace', "base", 'saveFigures', true, 'showFigures', false, ...
        'outputDir', ddpg_folder, 'filePrefix', "DDPG", 'figureFormat', "png", ...
        'closeAfterSave', true, 'timestamp', timestamp);
    if ~isempty(trainingResults), viz_options.trainingResults = trainingResults; end
    visualization(viz_options);
    fprintf('✓ Visualization stored at: %s\n', ddpg_folder);
end
