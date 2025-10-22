function [agent, results] = train_model(env, initial_agent, options)
if nargin < 2
    initial_agent = [];
end
if nargin < 3 || isempty(options)
    options = struct();
end

fprintf('\n=== DDPG Training ===\n');

obs_info = getObservationInfo(env);
act_info = getActionInfo(env);

maxEpisodes = get_option(options, 'maxEpisodes', 2000);
maxSteps = get_option(options, 'maxSteps', 720);
stopValue = get_option(options, 'stopValue', Inf);

if isa(initial_agent, 'rl.agent.rlDDPGAgent')
    obs_match = isequal(initial_agent.ObservationInfo.Dimension, obs_info.Dimension);
    act_match = isequal(initial_agent.ActionInfo.Dimension, act_info.Dimension);
    if obs_match && act_match
        agent = initial_agent;
        fprintf('✓ Continue training existing DDPG agent\n');
    else
        fprintf('⚠ Existing agent mismatch, recreating\n');
        agent = create_ddpg_agent(obs_info, act_info, env, options);
    end
else
    agent = create_ddpg_agent(obs_info, act_info, env, options);
end

% 根据当前动作范围重新配置噪声，防止OU噪声不稳定
action_range = act_info.UpperLimit - act_info.LowerLimit;
action_scale = action_range / 2;
sample_time_config = [];
try
    sample_time_config = agent.AgentOptions.SampleTime;
catch
end
if isempty(sample_time_config) || ~isnumeric(sample_time_config) || sample_time_config <= 0
    sample_time_config = infer_environment_sample_time(env, 3600);
end

agent.AgentOptions = configure_agent_noise(agent.AgentOptions, action_scale, sample_time_config, '继续训练');

train_opts = rlTrainingOptions;
train_opts.MaxEpisodes = maxEpisodes;
train_opts.MaxStepsPerEpisode = maxSteps;
train_opts.Plots = 'training-progress';
train_opts.ScoreAveragingWindowLength = max(1, min(5, maxEpisodes));
train_opts.Verbose = true;

if isfinite(stopValue)
    train_opts.StopTrainingCriteria = 'AverageReward';
    train_opts.StopTrainingValue = stopValue;
else
    train_opts.StopTrainingCriteria = 'EpisodeCount';
    train_opts.StopTrainingValue = maxEpisodes;
end

fprintf('\n=== Pre-training validation ===\n');
verify_continuous_action_output(agent, obs_info, act_info, 'Pre-training');

ensure_run_manager_on_path();
run_best_manager('init');

t_start = tic;
if ~evalin('base', 'exist(''g_episode_num'', ''var'')')
    assignin('base', 'g_episode_num', 0);
end

try
    saveBestFlag = get_option(options, 'saveBestDuringTraining', false);
    if saveBestFlag
        [agent, results] = train_with_online_best_saving(agent, env, train_opts, maxEpisodes, maxSteps);
        train_time = results.training_time;
    else
        stats = train(agent, env, train_opts);
        train_time = toc(t_start);
        results = summarize_training(stats, train_time, maxEpisodes);
    end
    fprintf('\n✓ DDPG training finished\n');
    fprintf('  - Episodes: %d\n', results.total_episodes);
    fprintf('  - Best reward: %.1f\n', results.best_reward);
catch ME
    train_time = toc(t_start);
    fprintf('\n✗ DDPG training failed: %s\n', ME.message);
    fprintf('  - Training duration: %.1f s\n', train_time);
    rethrow(ME);
end

fprintf('\n=== Post-training validation ===\n');
verify_continuous_action_output(agent, obs_info, act_info, 'Post-training');

try
    fprintf('\n=== Best episode extraction ===\n');
    episodeData = extract_best_episode([], agent, env, maxSteps);
    run_best_manager('save', agent, results, episodeData);
catch ME
    fprintf('⚠ Best episode save failed: %s\n', ME.message);
end
end

function agent = create_ddpg_agent(obs_info, act_info, env, options)
variant = lower(get_option(options, 'agentVariant', 'advanced'));
switch variant
    case {'simple'}
        agent = create_simple_agent(obs_info, act_info, env, options);
    case {'legacy'}
        agent = create_legacy_agent(obs_info, act_info, env, options);
    otherwise
        agent = create_advanced_agent(obs_info, act_info, env, options);
end
end

%% 创建高级智能体 (修正版，最佳性能)
function agent = create_advanced_agent(obs_info, act_info, env, options)
    if nargin < 4 || isempty(options)
        options = struct();
    end

    fprintf('\n创建高级DDPG智能体...\n');

    num_obs = obs_info.Dimension(1);
    num_act = act_info.Dimension(1);

    actor_layers_setting = get_option(options, 'actorLayerSizes', [256 128 64]);
    critic_state_layers = get_option(options, 'criticStateLayerSizes', [256 128]);
    critic_common_layers = get_option(options, 'criticCommonLayerSizes', [128 64]);

    action_range = act_info.UpperLimit - act_info.LowerLimit;
    action_scale = action_range / 2;
    action_bias = (act_info.UpperLimit + act_info.LowerLimit) / 2;
    sample_time = get_option(options, 'sampleTime', infer_environment_sample_time(env, 3600));
    fprintf('  - 采样时间: %.6f s\n', sample_time);

    %% 1. 创建Critic网络（状态-动作融合架构）
    % 状态处理分支
    state_layers = [
        featureInputLayer(num_obs, 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(critic_state_layers(1), 'Name', 'state_fc1')
        reluLayer('Name', 'state_relu1')
        fullyConnectedLayer(critic_state_layers(min(end, numel(critic_state_layers))), 'Name', 'state_fc2')
        reluLayer('Name', 'state_relu2')
    ];

    % 动作处理分支
    action_layers = [
        featureInputLayer(num_act, 'Normalization', 'none', 'Name', 'action')
        fullyConnectedLayer(128, 'Name', 'action_fc')
        reluLayer('Name', 'action_relu')
    ];

    % 创建网络图并连接
    critic_graph = layerGraph(state_layers);
    critic_graph = addLayers(critic_graph, action_layers);
    critic_graph = addLayers(critic_graph, [
        additionLayer(2, 'Name', 'state_action_add')
        fullyConnectedLayer(critic_common_layers(1), 'Name', 'common_fc1')
        reluLayer('Name', 'common_relu1')
        fullyConnectedLayer(critic_common_layers(min(end, numel(critic_common_layers))), 'Name', 'common_fc2')
        reluLayer('Name', 'common_relu2')
        fullyConnectedLayer(1, 'Name', 'q_value_output')
    ]);

    critic_graph = connectLayers(critic_graph, 'state_relu2', 'state_action_add/in1');
    critic_graph = connectLayers(critic_graph, 'action_relu', 'state_action_add/in2');

    critic_options = rlRepresentationOptions(...
        'LearnRate', 1e-3, ...
        'GradientThreshold', 1, ...
        'UseDevice', 'cpu');

    critic = rlQValueRepresentation(critic_graph, obs_info, act_info, ...
        'Observation', {'state'}, 'Action', {'action'}, critic_options);

    %% 2. 创建Actor网络（确保连续输出）
    % 关键修复：正确配置scalingLayer
    actor_layers = [
        featureInputLayer(num_obs, 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(actor_layers_setting(1), 'Name', 'actor_fc1')
        reluLayer('Name', 'actor_relu1')
        dropoutLayer(0.1, 'Name', 'dropout1')
        fullyConnectedLayer(actor_layers_setting(min(end, numel(actor_layers_setting))), 'Name', 'actor_fc2')
        reluLayer('Name', 'actor_relu2')
        dropoutLayer(0.1, 'Name', 'dropout2')
        fullyConnectedLayer(actor_layers_setting(max(1, min(end, numel(actor_layers_setting)))), 'Name', 'actor_fc3')
        reluLayer('Name', 'actor_relu3')
        fullyConnectedLayer(num_act, 'Name', 'pre_action')
        tanhLayer('Name', 'tanh_activation')  % 输出[-1,1]
        scalingLayer('Name', 'action_scaling', 'Scale', action_scale, 'Bias', action_bias)
    ];

    actor_options = rlRepresentationOptions(...
        'LearnRate', 5e-4, ...
        'GradientThreshold', 1, ...
        'UseDevice', 'cpu');

    actor = rlDeterministicActorRepresentation(actor_layers, obs_info, act_info, ...
        'Observation', {'state'}, actor_options);

    %% 3. 创建DDPG智能体（优化配置）
    agent_options = rlDDPGAgentOptions(...
        'SampleTime', sample_time, ...
        'TargetSmoothFactor', get_option(options, 'targetSmoothFactor', 1e-3), ...
        'DiscountFactor', get_option(options, 'discountFactor', 0.99), ...
        'MiniBatchSize', get_option(options, 'miniBatchSize', 64), ...
        'ExperienceBufferLength', get_option(options, 'experienceBufferLength', 1e6));

    agent_options = configure_agent_noise(agent_options, action_scale, sample_time, '高级');

    agent = rlDDPGAgent(actor, critic, agent_options);

    fprintf('✓ 高级DDPG智能体创建完成\n');
    fprintf('  - Actor网络: [256→128→64] + Scaling(%.1f, %.1f)\n', action_scale, action_bias);
    fprintf('  - Critic网络: 状态-动作融合架构\n');
end

%% 创建简化智能体 (兼容版，最大兼容性)
function agent = create_simple_agent(obs_info, act_info, env, options)
    if nargin < 4
        options = struct();
    end
    fprintf('\n创建简化DDPG智能体...\n');

    num_obs = obs_info.Dimension(1);
    num_act = act_info.Dimension(1);

    % 计算缩放参数
    action_range = act_info.UpperLimit - act_info.LowerLimit;
    action_scale = action_range / 2;
    action_bias = (act_info.UpperLimit + act_info.LowerLimit) / 2;

    sample_time = infer_environment_sample_time(env, 3600);
    fprintf('  - 采样时间: %.6f s\n', sample_time);

    %% 1. 简化Actor网络
    actor_layers = [
        featureInputLayer(num_obs, 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(128, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(64, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(num_act, 'Name', 'fc3')
        tanhLayer('Name', 'tanh')
        scalingLayer('Name', 'scaling', 'Scale', action_scale, 'Bias', action_bias)
    ];

    actor_options = rlRepresentationOptions('LearnRate', 1e-3);
    actor = rlDeterministicActorRepresentation(actor_layers, obs_info, act_info, ...
        'Observation', {'state'}, actor_options);

    %% 2. 简化Critic网络
    state_layers = [
        featureInputLayer(num_obs, 'Name', 'state')
        fullyConnectedLayer(64, 'Name', 'state_fc')
        reluLayer('Name', 'state_relu')
    ];

    action_layers = [
        featureInputLayer(num_act, 'Name', 'action')
        fullyConnectedLayer(64, 'Name', 'action_fc')
    ];

    critic_graph = layerGraph();
    critic_graph = addLayers(critic_graph, state_layers);
    critic_graph = addLayers(critic_graph, action_layers);
    critic_graph = addLayers(critic_graph, [
        additionLayer(2, 'Name', 'add')
        reluLayer('Name', 'common_relu')
        fullyConnectedLayer(32, 'Name', 'common_fc')
        reluLayer('Name', 'common_relu2')
        fullyConnectedLayer(1, 'Name', 'q_output')
    ]);

    critic_graph = connectLayers(critic_graph, 'state_relu', 'add/in1');
    critic_graph = connectLayers(critic_graph, 'action_fc', 'add/in2');

    critic_options = rlRepresentationOptions('LearnRate', 1e-3);
    critic = rlQValueRepresentation(critic_graph, obs_info, act_info, ...
        'Observation', {'state'}, 'Action', {'action'}, critic_options);

    %% 3. 基础DDPG智能体
    agent_options = rlDDPGAgentOptions();
    agent_options.SampleTime = get_option(options, 'sampleTime', sample_time);
    agent_options.DiscountFactor = get_option(options, 'discountFactor', 0.99);
    agent_options.MiniBatchSize = get_option(options, 'miniBatchSize', 32);
    agent_options.ExperienceBufferLength = get_option(options, 'experienceBufferLength', 5e4);

    try
        agent_options.TargetSmoothFactor = 1e-3;
    catch
        % 忽略不支持的参数
    end

    agent_options = configure_agent_noise(agent_options, action_scale, sample_time, '简化');

    agent = rlDDPGAgent(actor, critic, agent_options);

    fprintf('✓ 简化DDPG智能体创建完成\n');
    fprintf('  - Actor网络: [128→64] + Scaling(%.1f, %.1f)\n', action_scale, action_bias);
    fprintf('  - Critic网络: 简化融合架构\n');
end

%% 创建原始智能体 (保持原有逻辑)
function agent = create_legacy_agent(obs_info, act_info, env, options)
    if nargin < 4
        options = struct();
    end
    fprintf('\n创建原始DDPG智能体...\n');

    num_obs = obs_info.Dimension(1);
    num_act = act_info.Dimension(1);

    action_range = act_info.UpperLimit - act_info.LowerLimit;
    action_scale = action_range / 2;
    sample_time = infer_environment_sample_time(env, 3600);
    fprintf('  - 采样时间: %.6f s\n', sample_time);

    %% 1. 原始Critic网络
    critic_layer_sizes = [128, 64];

    state_path = [
        featureInputLayer(num_obs, 'Normalization', 'none', 'Name', 'observation')
        fullyConnectedLayer(critic_layer_sizes(1), 'Name', 'CriticStateFC1')
        reluLayer('Name', 'CriticStateRelu1')
    ];

    action_path = [
        featureInputLayer(num_act, 'Normalization', 'none', 'Name', 'action')
        fullyConnectedLayer(critic_layer_sizes(1), 'Name', 'CriticActionFC1')
    ];

    common_path = [
        additionLayer(2, 'Name', 'add')
        reluLayer('Name', 'CriticCommonRelu1')
        fullyConnectedLayer(critic_layer_sizes(2), 'Name', 'CriticFC2')
        reluLayer('Name', 'CriticRelu2')
        fullyConnectedLayer(1, 'Name', 'CriticOutput')
    ];

    critic_network = layerGraph(state_path);
    critic_network = addLayers(critic_network, action_path);
    critic_network = addLayers(critic_network, common_path);
    critic_network = connectLayers(critic_network, 'CriticStateRelu1', 'add/in1');
    critic_network = connectLayers(critic_network, 'CriticActionFC1', 'add/in2');

    critic_options = rlRepresentationOptions(...
        'LearnRate', 1e-3, ...
        'GradientThreshold', 1, ...
        'UseDevice', 'cpu');

    critic = rlQValueRepresentation(critic_network, obs_info, act_info, ...
        'Observation', {'observation'}, 'Action', {'action'}, critic_options);

    %% 2. 原始Actor网络（注意：这里有bug，但保持原有逻辑）
    actor_layer_sizes = [128, 64];

    actor_network = [
        featureInputLayer(num_obs, 'Normalization', 'none', 'Name', 'observation')
        fullyConnectedLayer(actor_layer_sizes(1), 'Name', 'ActorFC1')
        reluLayer('Name', 'ActorRelu1')
        fullyConnectedLayer(actor_layer_sizes(2), 'Name', 'ActorFC2')
        reluLayer('Name', 'ActorRelu2')
        fullyConnectedLayer(num_act, 'Name', 'ActorFC3')
        tanhLayer('Name', 'ActorTanh')
        % 原始版本的错误配置（仅用于对比）
        scalingLayer('Name', 'ActorScaling', 'Scale', act_info.UpperLimit)
    ];

    actor_options = rlRepresentationOptions(...
        'LearnRate', 5e-4, ...
        'GradientThreshold', 1, ...
        'UseDevice', 'cpu');

    actor = rlDeterministicActorRepresentation(actor_network, obs_info, act_info, ...
        'Observation', {'observation'}, actor_options);

    %% 3. 原始DDPG智能体
    agent_options = rlDDPGAgentOptions(...
        'SampleTime', sample_time, ...
        'TargetSmoothFactor', 1e-3, ...
        'DiscountFactor', 0.99, ...
        'MiniBatchSize', 64, ...
        'ExperienceBufferLength', 1e6);

    agent_options = configure_agent_noise(agent_options, action_scale, sample_time, '原始', 'std_gain', 0.08, 'decay_rate', 5e-5);

    agent = rlDDPGAgent(actor, critic, agent_options);

    fprintf('✓ 原始DDPG智能体创建完成\n');
    fprintf('  - 注意: 使用原始配置，可能存在已知问题\n');
end

%% 推断环境采样时间
function sample_time = infer_environment_sample_time(env, default_value)
    if nargin < 2 || isempty(default_value)
        default_value = 3600;
    end
    sample_time = default_value;
    if isempty(env)
        return;
    end
    candidate_props = {'Ts', 'AgentSampleTime', 'SampleTime'};
    for idx = 1:numel(candidate_props)
        prop = candidate_props{idx};
        if isprop(env, prop)
            value = env.(prop);
            if isnumeric(value) && isscalar(value) && value > 0
                sample_time = value;
                return;
            end
        end
    end
    try
        if isprop(env, 'Model') && isprop(env, 'AgentBlock')
            model_name = env.Model;
            agent_block = env.AgentBlock;
            block_path = agent_block;
            if isstring(block_path)
                block_path = char(block_path);
            end
            if isstring(model_name)
                model_name = char(model_name);
            end
            if ~contains(block_path, '/')
                block_path = [model_name '/' block_path];
            end
            param = get_param(block_path, 'SampleTime');
            numeric_dt = str2double(param);
            if ~isnan(numeric_dt) && numeric_dt > 0
                sample_time = numeric_dt;
                return;
            end
        end
    catch
        % 忽略无法获取采样时间的情况，保持默认值
    end
end

%% 配置探索噪声，确保OU噪声稳定
function agent_options = configure_agent_noise(agent_options, action_scale, sample_time, mode_label, varargin)
    if nargin < 4 || isempty(mode_label)
        mode_label = '';
    end
    prefix = '';
    if ~isempty(mode_label)
        prefix = ['[' mode_label '] '];
    end

    std_gain = 0.08;
    decay_rate = 5e-5; % 减缓噪声衰减速度，保持中后期探索能力
    base_theta = 0.15;
    for idx = 1:2:numel(varargin)
        key = varargin{idx};
        if idx+1 > numel(varargin)
            break;
        end
        value = varargin{idx+1};
        switch lower(key)
            case 'std_gain'
                if isnumeric(value) && isscalar(value) && value > 0
                    std_gain = value;
                end
            case 'decay_rate'
                if isnumeric(value) && isscalar(value) && value > 0
                    decay_rate = value;
                end
            case 'base_theta'
                if isnumeric(value) && isscalar(value) && value > 0
                    base_theta = value;
                end
        end
    end

    if ~(isnumeric(sample_time) && isscalar(sample_time) && sample_time > 0)
        sample_time = agent_options.SampleTime;
        if isempty(sample_time) || ~isnumeric(sample_time) || sample_time <= 0
            sample_time = 3600;
        end
    end

    noise_std = max(action_scale * std_gain, eps);
    max_theta = 2 / max(sample_time, eps);
    adjusted_theta = min(base_theta, 0.9 * max_theta);
    adjusted_theta = max(adjusted_theta, 1e-6);
    stability_metric = abs(1 - adjusted_theta * sample_time);

    if adjusted_theta < base_theta
        fprintf('  %sOU噪声θ从%.4g调整为%.6g以匹配采样时间%.4g秒，|1-θΔt|=%.4g\n', prefix, base_theta, adjusted_theta, sample_time, stability_metric);
    else
        fprintf('  %sOU噪声稳定性检查: θ=%.4g, 采样时间=%.4g秒, |1-θΔt|=%.4g\n', prefix, adjusted_theta, sample_time, stability_metric);
    end

    if exist('rlOrnsteinUhlenbeckActionNoise', 'file')
        try
            noise_obj = rlOrnsteinUhlenbeckActionNoise(...
                'Mean', 0, ...
                'StandardDeviation', noise_std, ...
                'MeanAttractionConstant', adjusted_theta, ...
                'StandardDeviationDecayRate', decay_rate);
            agent_options.NoiseOptions = noise_obj;
            fprintf('  %s使用OU噪声 (标准差 %.4g)\n', prefix, noise_std);
            return;
        catch ME
            fprintf('  %sOU噪声创建失败: %s，改用高斯噪声。\n', prefix, ME.message);
        end
    end

    try
        agent_options.NoiseOptions.Mean = 0;
        agent_options.NoiseOptions.MeanAttractionConstant = adjusted_theta;
        agent_options.NoiseOptions.Variance = noise_std^2;
        agent_options.NoiseOptions.VarianceDecayRate = decay_rate;
    catch
        agent_options.NoiseOptions = struct(...
            'Mean', 0, ...
            'MeanAttractionConstant', adjusted_theta, ...
            'Variance', noise_std^2, ...
            'VarianceDecayRate', decay_rate);
    end
    fprintf('  %s使用高斯噪声 (标准差 %.4g)\n', prefix, noise_std);
end

%% 配置训练选项
function train_opts = configure_training_options(mode)
    switch lower(mode)
        case 'advanced'
            train_opts = rlTrainingOptions(...
                'MaxEpisodes', 100, ...
                'MaxStepsPerEpisode', 720, ...
                'ScoreAveragingWindowLength', 5, ...
                'Verbose', true, ...
                'Plots', 'training-progress', ...
                'StopTrainingCriteria', 'AverageReward', ...
                'StopTrainingValue', 300, ...
                'SaveAgentCriteria', 'EpisodeReward', ...
                'SaveAgentValue', 200);

        case 'simple'
            train_opts = rlTrainingOptions();
            train_opts.MaxEpisodes = 50;
            train_opts.MaxStepsPerEpisode = 720;
            train_opts.Verbose = true;
            train_opts.Plots = 'training-progress';
            try
                train_opts.ScoreAveragingWindowLength = 5;
                train_opts.StopTrainingCriteria = 'AverageReward';
                train_opts.StopTrainingValue = 200;
            catch
                % 忽略不支持的选项
            end

        case 'legacy'
            train_opts = rlTrainingOptions(...
                'MaxEpisodes', 500, ...
                'MaxStepsPerEpisode', 720, ...
                'ScoreAveragingWindowLength', 5, ...
                'Verbose', true, ...
                'Plots', 'training-progress', ...
                'StopTrainingCriteria', 'AverageReward', ...
                'StopTrainingValue', 500, ...
                'SaveAgentCriteria', 'EpisodeReward', ...
                'SaveAgentValue', 300);

        otherwise
            error('未知训练模式: %s', mode);
    end

    fprintf('训练配置: %d回合, %d步/回合\n', train_opts.MaxEpisodes, train_opts.MaxStepsPerEpisode);
end

%% 处理训练结果
function training_results = process_training_results(training_stats, training_time, mode)
    training_results = struct();
    training_results.mode = mode;
    training_results.total_episodes = length(training_stats.EpisodeReward);
    training_results.episode_rewards = training_stats.EpisodeReward;
    training_results.training_time = training_time;
    training_results.final_reward = training_stats.EpisodeReward(end);
    training_results.best_reward = max(training_stats.EpisodeReward);

    if isfield(training_stats, 'AverageReward')
        training_results.average_reward = training_stats.AverageReward(end);
    else
        training_results.average_reward = mean(training_stats.EpisodeReward);
    end

    if isfield(training_stats, 'EpisodeSteps')
        training_results.episode_steps = training_stats.EpisodeSteps;
    end
end

%% 显示训练总结
function display_training_summary(results, mode)
    fprintf('=== 训练总结 (%s模式) ===\n', upper(mode));
    fprintf('总回合数: %d\n', results.total_episodes);
    fprintf('最终奖励: %.1f\n', results.final_reward);
    fprintf('最佳奖励: %.1f\n', results.best_reward);
    fprintf('平均奖励: %.1f\n', results.average_reward);
    fprintf('训练时长: %.1f分钟\n', results.training_time/60);
    fprintf('================================\n');
end

%% 验证连续动作输出
function verify_continuous_action_output(agent, obs_info, act_info, phase)
    try
        fprintf('%s连续动作验证:\n', phase);

        % 关闭探索噪声
        agent.UseExplorationPolicy = false;

        % 生成多样化测试观测
        num_tests = 15;
        actions = zeros(num_tests, 1);

        for i = 1:num_tests
            % 生成随机但合理的观测
            test_obs = rand(obs_info.Dimension) * 0.6 + 0.2;  % [0.2, 0.8]范围
            action_cell = getAction(agent, {test_obs});
            actions(i) = action_cell{1};
        end

        % 分析结果
        action_min = min(actions);
        action_max = max(actions);
        action_std = std(actions);
        unique_count = length(unique(round(actions, 1)));

        fprintf('  - 动作范围: [%.1f, %.1f] W\n', action_min, action_max);
        fprintf('  - 标准差: %.1f W\n', action_std);
        fprintf('  - 唯一值: %d/%d\n', unique_count, num_tests);

        % 验证条件
        in_range = all(actions >= act_info.LowerLimit-100) && all(actions <= act_info.UpperLimit+100);
        has_variation = action_std > 10;
        is_continuous = unique_count > num_tests * 0.6;

        if in_range && has_variation && is_continuous
            fprintf('  ✅ 连续动作输出正常\n');
        else
            fprintf('  ⚠️ 连续动作输出异常:\n');
            if ~in_range
                fprintf('    - 动作超出范围 [%.0f, %.0f]\n', act_info.LowerLimit, act_info.UpperLimit);
            end
            if ~has_variation
                fprintf('    - 动作变化不足 (标准差: %.1f)\n', action_std);
            end
            if ~is_continuous
                fprintf('    - 连续性不足 (唯一值: %d/%d)\n', unique_count, num_tests);
            end
        end
    catch ME
        fprintf('  ❌ 验证失败: %s\n', ME.message);
    end
end

function ensure_run_manager_on_path()
% 
persistent path_initialized
if isempty(path_initialized)
    current_dir = fileparts(mfilename('fullpath'));
    project_root = find_project_root(current_dir);
    run_manager_dir = fullfile(project_root, 'matlab', 'src', 'run_manager');
    if exist(run_manager_dir, 'dir')
        addpath(run_manager_dir);
    end
    path_initialized = true;
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
error('DDPG:ProjectRootNotFound', '  %s', start_dir);
end

function [agent, results] = train_with_online_best_saving(agent, env, base_opts, maxEpisodes, maxSteps)
if nargin < 4 || isempty(maxEpisodes)
    maxEpisodes = base_opts.MaxEpisodes;
end
if nargin < 5 || isempty(maxSteps)
    maxSteps = base_opts.MaxStepsPerEpisode;
end

bestReward = -inf;
allRewards = [];
t_start_local = tic;
assignin('base','g_episode_num',0);

for ep = 1:maxEpisodes
    assignin('base','g_episode_num', ep);
    fprintf('[ResetFcn] Episode %d reset at %s\n', ep, char(datetime('now','Format','HH:mm:ss')));

    opts = base_opts;
    try
        opts.StopTrainingCriteria = 'EpisodeCount';
    catch
    end
    opts.StopTrainingValue = 1;
    opts.Plots = 'none';

    stats = train(agent, env, opts);
    if isfield(stats, 'EpisodeReward') && ~isempty(stats.EpisodeReward)
        r = double(stats.EpisodeReward(end));
    else
        r = NaN;
    end
    allRewards(end+1) = r; %#ok<AGROW>

    if r > bestReward
        bestReward = r;
        try
            episodeData = extract_best_episode([], agent, env, maxSteps);
            avg_reward = compute_average_reward(allRewards);
            tmp = struct('episode_rewards', allRewards, ...
                         'best_reward', bestReward, ...
                         'total_episodes', numel(allRewards), ...
                         'average_reward', avg_reward, ...
                         'training_time', toc(t_start_local));
            run_best_manager('save', agent, tmp, episodeData);
        catch ME
            fprintf('⚠ DDPG online save failed: %s\n', ME.message);
        end
    end
end

avg_reward = compute_average_reward(allRewards);
results = struct('episode_rewards', allRewards, ...
                 'best_reward', bestReward, ...
                 'total_episodes', numel(allRewards), ...
                 'average_reward', avg_reward, ...
                 'training_time', toc(t_start_local));
end

function results = summarize_training(stats, train_time, maxEpisodes)
results = struct();

if isfield(stats, 'EpisodeReward') && ~isempty(stats.EpisodeReward)
    rewards = double(stats.EpisodeReward(:));
else
    rewards = [];
end

results.episode_rewards = rewards;
results.total_episodes = numel(rewards);
results.training_time = train_time;
results.max_episodes_config = maxEpisodes;

if isempty(rewards)
    results.best_reward = -inf;
    results.average_reward = NaN;
    results.final_reward = NaN;
else
    results.best_reward = max(rewards);
    results.average_reward = compute_average_reward(rewards);
    results.final_reward = rewards(end);
end

if isfield(stats, 'EpisodeSteps')
    results.episode_steps = stats.EpisodeSteps;
end
if isfield(stats, 'AverageReward') && ~isempty(stats.AverageReward)
    results.average_reward_window = stats.AverageReward;
end
end

function value = get_option(options, field, default_value)
if isstruct(options) && isfield(options, field)
    value = options.(field);
else
    value = default_value;
end
end

function avg = compute_average_reward(rewards)
if isempty(rewards)
    avg = NaN;
    return;
end
valid = rewards(~isnan(rewards));
if isempty(valid)
    avg = NaN;
else
    avg = mean(valid);
end
end
