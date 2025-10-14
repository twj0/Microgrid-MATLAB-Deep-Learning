function main(options)
    if nargin < 1 || isempty(options)
        options = struct();
    end

    clc;
    close all;

    fprintf('========================================\n');
    fprintf('  微电网SAC强化学习训练系统 v1.0\n');
    fprintf('========================================\n');
    fprintf('启动时间: %s\n', char(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss")));

    try
        config = initialize_environment(options);
        fprintf('✓ 环境路径初始化完成\n');

        try
            load_fuzzy_logic_system(config.model_dir);
            fprintf('✓ 模糊逻辑系统加载完成\n');
        catch ME
            fprintf('⚠ 模糊逻辑系统加载失败: %s\n', ME.message);
        end

        [data, raw_data] = load_and_validate_data(config);
        fprintf('  - 光伏数据: %.0f 小时 (%d 样本)\n', data.duration_hours, data.sample_count);
        fprintf('  - 负载数据: %.0f 小时 (%d 样本)\n', ...
            calculate_duration_hours(data.load_power), calculate_sample_count(data.load_power));
        fprintf('  - 电价数据: %.0f 小时 (%d 样本)\n', ...
            calculate_duration_hours(data.price_profile), calculate_sample_count(data.price_profile));
        push_simulation_data_to_base(data, raw_data);
        fprintf('✓ 数据加载与分发完成\n');

        [env, agent_var_name] = setup_simulink_environment(config);
        fprintf('✓ Simulink环境配置完成\n');

        initial_agent = [];
        if ~isempty(agent_var_name)
            try
                initial_agent = evalin('base', agent_var_name);
            catch
                initial_agent = [];
            end
        end

        [agent, results] = train_model(env, initial_agent, config.training);
        fprintf('✓ SAC训练完成\n');

        save_results(agent, results, config, agent_var_name);
        fprintf('✓ 训练结果已保存\n');

        verify_policy(agent, env);
        fprintf('✓ 策略验证完成\n');

    catch ME
        fprintf('\n✗ 发生错误: %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('位置: %s (第%d行)\n', ME.stack(1).name, ME.stack(1).line);
        end
        rethrow(ME);
    end

    fprintf('========================================\n');
    fprintf('  训练流程结束\n');
    fprintf('========================================\n');
end

function config = initialize_environment(options)
    project_root = fileparts(fileparts(pwd));
    model_dir = fullfile(project_root, 'model');
    addpath(genpath(project_root));
    addpath(model_dir);

    config.model_name = 'Microgrid';
    config.model_dir = model_dir;
    config.model_path = fullfile(model_dir, [config.model_name '.slx']);
    config.data_path = fullfile(project_root, 'matlab', 'src', 'microgrid_simulation_data.mat');
    config.simulation_days = get_option(options, 'simulation_days', 30);
    config.sample_time = 3600;
    config.simulation_time = config.simulation_days * 24 * config.sample_time;

    config.training.maxEpisodes = get_option(options, 'maxEpisodes', 120);
    config.training.maxSteps = get_option(options, 'maxSteps', 720);
    config.training.stopValue = get_option(options, 'stopValue', 250);
    config.training.sampleTime = config.sample_time;

    if ~exist(config.model_path, 'file')
        error('未找到Simulink模型: %s', config.model_path);
    end
    if ~exist(config.data_path, 'file')
        error('未找到数据文件: %s\n请先运行 matlab/src/generate_data.m', config.data_path);
    end
end

function [data, raw_data] = load_and_validate_data(config)
    raw_data = load(config.data_path);

    required_vars = {'pv_power_profile', 'load_power_profile'};
    for k = 1:numel(required_vars)
        if ~isfield(raw_data, required_vars{k})
            error('缺少必要数据: %s', required_vars{k});
        end
    end

    if isfield(raw_data, 'price_profile')
        price_profile = raw_data.price_profile;
    elseif isfield(raw_data, 'price_data')
        price_profile = raw_data.price_data;
    else
        error('缺少电价数据: price_profile 或 price_data');
    end

    if ~isa(price_profile, 'timeseries')
        if isfield(raw_data, 'time_seconds')
            time_vector = raw_data.time_seconds(:);
        else
            time_vector = (0:numel(price_profile)-1)' * config.sample_time;
        end
        price_ts = timeseries(price_profile(:), time_vector);
        price_ts.Name = 'Electricity Price Profile';
        price_ts.DataInfo.Units = 'CNY/kWh';
    else
        price_ts = price_profile;
    end

    data.price_profile = price_ts;
    data.pv_power = raw_data.pv_power_profile;
    data.load_power = raw_data.load_power_profile;
    data.duration_hours = calculate_duration_hours(data.pv_power);
    data.sample_count = calculate_sample_count(data.pv_power);

    required_samples = max(config.training.maxSteps, ceil(config.simulation_time / config.sample_time));

    if data.sample_count < required_samples
        error(['数据样本不足: 需要>=%d个样本（约%.0f小时）', ...
               '，当前仅有%d个样本（约%.0f小时）。\n', ...
               '请运行 matlab/src/generate_data.m 重新生成足够长度的时序数据。'], ...
            required_samples, required_samples * (config.sample_time/3600), ...
            data.sample_count, data.duration_hours);
    end
end

function [env, agent_var_name] = setup_simulink_environment(config)
    if ~bdIsLoaded(config.model_name)
        load_system(config.model_path);
    end
    set_param(config.model_name, 'StopTime', num2str(config.simulation_time));

    obs_info = rlNumericSpec([9 1]);
    obs_info.Name = 'Microgrid Observations';

    act_info = rlNumericSpec([1 1]);
    act_info.LowerLimit = -10e3;
    act_info.UpperLimit = 10e3;
    act_info.Name = 'Battery Power';

    agent_block = [config.model_name '/RL Agent'];
    agent_var_name = ensure_agent_variable(agent_block, obs_info, act_info, config.sample_time);

    env = rlSimulinkEnv(config.model_name, agent_block, obs_info, act_info);
    env.ResetFcn = @(in)setVariable(in, 'initial_soc', 50);
end

function save_results(agent, results, config, agent_var_name)
    if ~isempty(agent_var_name)
        assignin('base', agent_var_name, agent);
        fprintf('  - 工作区变量已更新: %s\n', agent_var_name);
    end

    timestamp = char(datetime("now", "Format", "yyyyMMdd_HHmmss"));
    filename = sprintf('sac_agent_%s.mat', timestamp);
    metadata = struct('config', config, 'timestamp', datetime("now"), 'matlab_version', version);
    save(filename, 'agent', 'results', 'metadata');
    fprintf('  - 已保存: %s\n', filename);
end

function verify_policy(agent, env)
    deterministic_supported = false;
    old_flag = [];
    try
        obs_info = getObservationInfo(env);
        act_info = getActionInfo(env);
        deterministic_supported = isprop(agent, 'UseDeterministicExploitationPolicy');
        if deterministic_supported
            old_flag = agent.UseDeterministicExploitationPolicy;
            agent.UseDeterministicExploitationPolicy = true;
        end

        samples = 20;
        actions = zeros(samples, 1);
        for i = 1:samples
            obs = rand(obs_info.Dimension) * 0.6 + 0.2;
            action_cell = getAction(agent, {obs});
            actions(i) = action_cell{1};
        end

        if deterministic_supported
            agent.UseDeterministicExploitationPolicy = old_flag;
        end

        fprintf('  - 验证动作范围: [%.2f, %.2f] kW\n', min(actions)/1000, max(actions)/1000);
        fprintf('  - 动作标准差: %.2f W\n', std(actions));

        if all(actions >= act_info.LowerLimit - 100) && all(actions <= act_info.UpperLimit + 100)
            fprintf('  ✅ 动作范围合法\n');
        else
            fprintf('  ⚠ 动作存在越界\n');
        end
    catch ME
        if deterministic_supported && ~isempty(old_flag)
            try
                agent.UseDeterministicExploitationPolicy = old_flag;
            catch
            end
        end
        fprintf('  ⚠ 策略验证失败: %s\n', ME.message);
    end
end

function load_fuzzy_logic_system(model_dir)
    fis_files = {'Battery_State_Type2.fis','Battery_State.fis','Economic_Reward.fis','Economic_Decision.fis'};
    for k = 1:numel(fis_files)
        path_fis = fullfile(model_dir, fis_files{k});
        if exist(path_fis, 'file')
            fis = readfis(path_fis);
            assignin('base', erase(fis_files{k}, '.fis'), fis);
        end
    end

    default_vars = {'price_norm',0.5; 'soc_current',50; 'soc_diff',0; 'reward_signal',0};
    for k = 1:size(default_vars,1)
        if ~evalin('base', sprintf('exist(''%s'',''var'')', default_vars{k,1}))
            assignin('base', default_vars{k,1}, default_vars{k,2});
        end
    end
    if ~evalin('base', 'exist(''eml_transient'',''var'')')
        assignin('base', 'eml_transient', struct());
    end
end

function push_simulation_data_to_base(data, raw_data)
    assignin('base', 'price_profile', data.price_profile);
    assignin('base', 'pv_power_profile', data.pv_power);
    assignin('base', 'load_power_profile', data.load_power);

    fields = fieldnames(raw_data);
    for k = 1:numel(fields)
        name = fields{k};
        if any(strcmp(name, {'price_profile','pv_power_profile','load_power_profile'}))
            continue;
        end
        assignin('base', name, raw_data.(name));
    end
end

function duration = calculate_duration_hours(ts)
    duration = 0;
    if ~isa(ts, 'timeseries') || isempty(ts.Data)
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

function agent_var_name = ensure_agent_variable(agent_block, obs_info, act_info, sample_time)
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
    try
        existing = evalin('base', agent_var_name);
        if isa(existing, 'rl.agent.Agent')
            return;
        end
    catch
    end

    try
        placeholder = create_placeholder_agent(obs_info, act_info, sample_time);
        assignin('base', agent_var_name, placeholder);
        fprintf('  - 已创建SAC占位智能体: %s\n', agent_var_name);
    catch ME
        fprintf('  - SAC占位智能体创建失败: %s\n', ME.message);
    end
end

function agent = create_placeholder_agent(obs_info, act_info, sample_time)
    cfg.actorSizes = [64 32];
    cfg.criticSizes = [128 64];
    cfg.sampleTime = sample_time;
    cfg.targetEntropy = -act_info.Dimension(1);
    agent = build_sac_agent(obs_info, act_info, cfg);
end

function agent = build_sac_agent(obs_info, act_info, cfg)
    actor = build_actor(obs_info, act_info, cfg.actorSizes);
    critic1 = build_critic(obs_info, act_info, cfg.criticSizes, 'c1');
    critic2 = build_critic(obs_info, act_info, cfg.criticSizes, 'c2');

    agent_opts = rlSACAgentOptions();
    agent_opts.SampleTime = cfg.sampleTime;
    agent_opts.TargetSmoothFactor = 5e-3;
    agent_opts.DiscountFactor = 0.99;
    agent_opts.ExperienceBufferLength = 1e6;
    agent_opts.MiniBatchSize = 128;
    agent_opts.EntropyWeightOptions.TargetEntropy = cfg.targetEntropy;

    call_attempts = {
        {actor, [critic1, critic2], agent_opts};
        {actor, [critic1, critic2]};
        {actor, critic1, critic2, agent_opts};
        {actor, critic1, agent_opts};
        {actor, critic1}
    };

    agent = [];
    last_error = [];
    for idx = 1:numel(call_attempts)
        args = call_attempts{idx};
        try
            agent = rlSACAgent(args{:});
            last_error = [];
            break;
        catch ME
            last_error = ME;
        end
    end

    if isempty(agent)
        rethrow(last_error);
    end
end

function actor = build_actor(obs_info, act_info, layer_sizes)
    num_obs = obs_info.Dimension(1);
    num_act = act_info.Dimension(1);
    action_range = act_info.UpperLimit - act_info.LowerLimit;
    action_scale = action_range / 2;
    action_bias = (act_info.UpperLimit + act_info.LowerLimit) / 2;

    lg = layerGraph();
    lg = addLayers(lg, featureInputLayer(num_obs, 'Normalization', 'none', 'Name', 'state'));
    prev = 'state';
    for i = 1:numel(layer_sizes)
        fc_name = sprintf('actor_fc%d', i);
        relu_name = sprintf('actor_relu%d', i);
        lg = addLayers(lg, fullyConnectedLayer(layer_sizes(i), 'Name', fc_name));
        lg = addLayers(lg, reluLayer('Name', relu_name));
        lg = connectLayers(lg, prev, fc_name);
        lg = connectLayers(lg, fc_name, relu_name);
        prev = relu_name;
    end

    mean_layer = fullyConnectedLayer(num_act, 'Name', 'action_mean_fc');
    logstd_layer = fullyConnectedLayer(num_act, 'Name', 'action_logstd_fc');
    lg = addLayers(lg, mean_layer);
    lg = addLayers(lg, logstd_layer);
    lg = connectLayers(lg, prev, 'action_mean_fc');
    lg = connectLayers(lg, prev, 'action_logstd_fc');

    scale_layer = scalingLayer('Name', 'action_mean_scale', 'Scale', action_scale, 'Bias', action_bias);
    lg = addLayers(lg, scale_layer);
    lg = connectLayers(lg, 'action_mean_fc', 'action_mean_scale');

    softplus_layer = softplusLayer('Name', 'action_std_softplus');
    lg = addLayers(lg, softplus_layer);
    lg = connectLayers(lg, 'action_logstd_fc', 'action_std_softplus');

    std_scale = scalingLayer('Name', 'action_std_scale', 'Scale', action_scale, 'Bias', 1e-3);
    lg = addLayers(lg, std_scale);
    lg = connectLayers(lg, 'action_std_softplus', 'action_std_scale');

    actor_opts = rlRepresentationOptions('LearnRate', 3e-4, 'GradientThreshold', 1, 'UseDevice', 'cpu');
    mean_name = 'action_mean_scale';
    std_name = 'action_std_scale';
    actor = instantiate_gaussian_actor(lg, obs_info, act_info, mean_name, std_name, actor_opts);
end

function critic = build_critic(obs_info, act_info, layer_sizes, suffix)
    num_obs = obs_info.Dimension(1);
    num_act = act_info.Dimension(1);
    state_path = [
        featureInputLayer(num_obs, 'Normalization', 'none', 'Name', ['state_' suffix])
        fullyConnectedLayer(layer_sizes(1), 'Name', ['state_fc1_' suffix])
        reluLayer('Name', ['state_relu1_' suffix])
    ];

    action_path = [
        featureInputLayer(num_act, 'Normalization', 'none', 'Name', ['action_' suffix])
        fullyConnectedLayer(layer_sizes(1), 'Name', ['action_fc1_' suffix])
    ];

    critic_lg = layerGraph(state_path);
    critic_lg = addLayers(critic_lg, action_path);
    critic_lg = addLayers(critic_lg, additionLayer(2, 'Name', ['add_' suffix]));
    prev = ['add_' suffix];
    critic_lg = connectLayers(critic_lg, ['state_relu1_' suffix], [prev '/in1']);
    critic_lg = connectLayers(critic_lg, ['action_fc1_' suffix], [prev '/in2']);

    for i = 2:numel(layer_sizes)
        fc_name = sprintf('critic_fc%d_%s', i, suffix);
        relu_name = sprintf('critic_relu%d_%s', i, suffix);
        critic_lg = addLayers(critic_lg, fullyConnectedLayer(layer_sizes(i), 'Name', fc_name));
        critic_lg = addLayers(critic_lg, reluLayer('Name', relu_name));
        critic_lg = connectLayers(critic_lg, prev, fc_name);
        critic_lg = connectLayers(critic_lg, fc_name, relu_name);
        prev = relu_name;
    end

    critic_lg = addLayers(critic_lg, fullyConnectedLayer(1, 'Name', ['q_value_' suffix]));
    critic_lg = connectLayers(critic_lg, prev, ['q_value_' suffix]);

    critic_opts = rlRepresentationOptions('LearnRate', 3e-4, 'GradientThreshold', 1, 'UseDevice', 'cpu');
    critic = instantiate_qvalue_representation(critic_lg, obs_info, act_info, ...
        ['state_' suffix], ['action_' suffix], critic_opts);
end

function actor = instantiate_gaussian_actor(lg, obs_info, act_info, mean_name, std_name, opts)
    if exist('rlContinuousGaussianActorRepresentation', 'file') == 2
        actor = rlContinuousGaussianActorRepresentation(lg, obs_info, act_info, ...
            'Observation', {'state'}, 'Action', {mean_name}, 'ActionStd', {std_name}, ...
            'Options', opts);
        return;
    end

    if exist('rlStochasticActorRepresentation', 'file') == 2
        call_attempts = {
            {'Observation', {'state'}, 'Action', {mean_name}, 'ActionStd', {std_name}, 'Options', opts};
            {'Observation', {'state'}, 'Action', {mean_name}, 'MeanOutputNames', {mean_name}, 'StdOutputNames', {std_name}, 'Options', opts};
            {'Observation', {'state'}, 'Action', {mean_name}, 'Options', opts};
            {opts}
        };
        for i = 1:numel(call_attempts)
            try
                args = call_attempts{i};
                actor = rlStochasticActorRepresentation(lg, obs_info, act_info, args{:});
                return;
            catch
            end
        end
        error('SAC:StochasticActorCreationFailed', '无法使用任何已知签名创建rlStochasticActorRepresentation。');
    end

    if exist('rlRepresentation', 'file') == 2
        call_attempts = {
            {'Observation', {'state'}, 'Action', {mean_name}, 'ActionStd', {std_name}, 'Options', opts};
            {'Observation', {'state'}, 'Action', {mean_name}, 'MeanOutputNames', {mean_name}, 'StdOutputNames', {std_name}, 'Options', opts};
            {'Observation', {'state'}, 'Action', {mean_name}, 'Options', opts};
            {opts}
        };
        for i = 1:numel(call_attempts)
            try
                args = call_attempts{i};
                actor = rlRepresentation(lg, obs_info, act_info, args{:});
                return;
            catch
            end
        end
    end

    error('SAC:MissingGaussianActorAPI', '当前MATLAB版本不支持创建连续高斯策略，请升级强化学习工具箱。');
end

function critic = instantiate_qvalue_representation(lg, obs_info, act_info, obs_name, act_name, opts)
    if exist('rlQValueRepresentation', 'file') == 2
        call_attempts = {
            {'Observation', {obs_name}, 'Action', {act_name}, 'Options', opts};
            {'Observation', {obs_name}, 'Action', {act_name}, opts};
            {'Observation', {obs_name}, 'Action', {act_name}};
            {opts}
        };
        for idx = 1:numel(call_attempts)
            try
                args = call_attempts{idx};
                critic = rlQValueRepresentation(lg, obs_info, act_info, args{:});
                return;
            catch
            end
        end
    end

    if exist('rlRepresentation', 'file') == 2
        call_attempts = {
            {'Observation', {obs_name}, 'Action', {act_name}, 'Options', opts};
            {'Observation', {obs_name}, 'Action', {act_name}, opts};
            {'Observation', {obs_name}, 'Action', {act_name}};
            {opts}
        };
        for idx = 1:numel(call_attempts)
            try
                args = call_attempts{idx};
                critic = rlRepresentation(lg, obs_info, act_info, args{:});
                return;
            catch
            end
        end
    end

    error('SAC:MissingQValueAPI', '当前MATLAB版本不支持创建Q-value表示，请升级强化学习工具箱。');
end

function value = get_option(options, field, default_value)
    if isstruct(options) && isfield(options, field)
        value = options.(field);
    else
        value = default_value;
    end
end
