function [agent, results] = train_model(env, initial_agent, options)
    if nargin < 2
        initial_agent = [];
    end
    if nargin < 3 || isempty(options)
        options = struct();
    end

    obs_info = getObservationInfo(env);
    act_info = getActionInfo(env);

    cfg.actorLayerSizes = get_option(options, 'actorLayerSizes', [256 128 64]);
    cfg.criticLayerSizes = get_option(options, 'criticLayerSizes', [256 128 64]);
    cfg.sampleTime = get_option(options, 'sampleTime', 3600);
    cfg.targetEntropy = get_option(options, 'targetEntropy', -act_info.Dimension(1));

    maxEpisodes = get_option(options, 'maxEpisodes', 120);
    maxSteps = get_option(options, 'maxSteps', 720);
    stopValue = get_option(options, 'stopValue', NaN);

    if isa(initial_agent, 'rl.agent.rlSACAgent')
        obs_match = isequal(initial_agent.ObservationInfo.Dimension, obs_info.Dimension);
        act_match = isequal(initial_agent.ActionInfo.Dimension, act_info.Dimension);
        if obs_match && act_match
            agent = initial_agent;
            fprintf('✓ 使用已有SAC智能体继续训练\n');
        else
            fprintf('⚠ 现有智能体与环境规格不匹配，将重新创建\n');
            agent = create_sac_agent(obs_info, act_info, cfg);
        end
    else
        agent = create_sac_agent(obs_info, act_info, cfg);
    end

    train_opts = rlTrainingOptions;
    train_opts.MaxEpisodes = maxEpisodes;
    train_opts.MaxStepsPerEpisode = maxSteps;
    train_opts.Plots = 'training-progress';
    train_opts.ScoreAveragingWindowLength = max(1, min(5, maxEpisodes));
    train_opts.Verbose = true;
    if ~isnan(stopValue)
        train_opts.StopTrainingCriteria = 'AverageReward';
        train_opts.StopTrainingValue = stopValue;
    end

    fprintf('\n=== SAC训练前验证 ===\n');
    verify_continuous_actions(agent, obs_info, act_info, '训练前');

    fprintf('\n=== 开始SAC训练 ===\n');
    t_start = tic;
    try
        stats = train(agent, env, train_opts);
        train_time = toc(t_start);
        training_summary = summarize_training(stats, train_time, maxEpisodes);
        results = training_summary;  % 将结果赋值给results变量
        fprintf('\n✓ SAC训练完成\n');
        fprintf('  - 总回合: %d\n', training_summary.total_episodes);
        fprintf('  - 最佳奖励: %.1f\n', training_summary.best_reward);
        fprintf('  - 平均奖励: %.1f\n', training_summary.average_reward);
    catch ME
        train_time = toc(t_start);
        fprintf('\n✗ SAC训练失败: %s\n', ME.message);
        fprintf('  - 训练持续时间: %.1f 秒\n', train_time);
        rethrow(ME);
    end

    fprintf('\n=== SAC训练后验证 ===\n');
    verify_continuous_actions(agent, obs_info, act_info, '训练后');
end

function agent = create_sac_agent(obs_info, act_info, cfg)
    actor = build_actor(obs_info, act_info, cfg.actorLayerSizes);
    critic1 = build_critic(obs_info, act_info, cfg.criticLayerSizes, 'c1');
    critic2 = build_critic(obs_info, act_info, cfg.criticLayerSizes, 'c2');

    agent_opts = rlSACAgentOptions();
    agent_opts.SampleTime = cfg.sampleTime;
    agent_opts.TargetSmoothFactor = 5e-3;
    agent_opts.DiscountFactor = 0.99;
    agent_opts.ExperienceBufferLength = 1e6;
    agent_opts.MiniBatchSize = 256;
    agent_opts.EntropyWeightOptions.TargetEntropy = cfg.targetEntropy;
    agent_opts.EntropyWeightOptions.LearnRate = 1e-3;

    call_attempts = {
        {actor, [critic1, critic2], agent_opts};
        {actor, [critic1, critic2]};
        {actor, critic1, critic2, agent_opts};
        {actor, critic1, agent_opts};
        {actor, critic1}
    };

    agent = [];
    last_error = [];
    for i = 1:numel(call_attempts)
        args = call_attempts{i};
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

    fprintf('✓ 创建SAC智能体成功\n');
    fprintf('  - Actor层级: %s\n', mat2str(cfg.actorLayerSizes));
    fprintf('  - Critic层级: %s\n', mat2str(cfg.criticLayerSizes));
end

function actor = build_actor(obs_info, act_info, layer_sizes)
    num_obs = obs_info.Dimension(1);
    num_act = act_info.Dimension(1);
    action_range = act_info.UpperLimit - act_info.LowerLimit;
    action_scale = action_range / 2;
    action_bias = (act_info.UpperLimit + act_info.LowerLimit) / 2;

    lg = layerGraph();
    lg = addLayers(lg, featureInputLayer(num_obs, 'Normalization', 'none', 'Name', 'state'));
    previous = 'state';
    for i = 1:numel(layer_sizes)
        fc = sprintf('actor_fc_%d', i);
        relu = sprintf('actor_relu_%d', i);
        lg = addLayers(lg, fullyConnectedLayer(layer_sizes(i), 'Name', fc));
        lg = addLayers(lg, reluLayer('Name', relu));
        lg = connectLayers(lg, previous, fc);
        lg = connectLayers(lg, fc, relu);
        previous = relu;
    end

    lg = addLayers(lg, fullyConnectedLayer(num_act, 'Name', 'actor_mean_fc'));
    lg = addLayers(lg, scalingLayer('Name', 'actor_mean_scale', 'Scale', action_scale, 'Bias', action_bias));
    lg = addLayers(lg, fullyConnectedLayer(num_act, 'Name', 'actor_std_fc'));
    lg = addLayers(lg, softplusLayer('Name', 'actor_std_softplus'));
    lg = addLayers(lg, scalingLayer('Name', 'actor_std_scale', 'Scale', action_scale, 'Bias', 1e-3));

    lg = connectLayers(lg, previous, 'actor_mean_fc');
    lg = connectLayers(lg, 'actor_mean_fc', 'actor_mean_scale');
    lg = connectLayers(lg, previous, 'actor_std_fc');
    lg = connectLayers(lg, 'actor_std_fc', 'actor_std_softplus');
    lg = connectLayers(lg, 'actor_std_softplus', 'actor_std_scale');

    opts = rlRepresentationOptions('LearnRate', 3e-4, 'GradientThreshold', 1, 'UseDevice', 'cpu');
    mean_name = 'actor_mean_scale';
    std_name = 'actor_std_scale';
    actor = instantiate_gaussian_actor(lg, obs_info, act_info, mean_name, std_name, opts);
end

function critic = build_critic(obs_info, act_info, layer_sizes, suffix)
    num_obs = obs_info.Dimension(1);
    num_act = act_info.Dimension(1);

    state_layers = [
        featureInputLayer(num_obs, 'Normalization', 'none', 'Name', ['state_' suffix])
        fullyConnectedLayer(layer_sizes(1), 'Name', ['state_fc1_' suffix])
        reluLayer('Name', ['state_relu1_' suffix])
    ];

    action_layers = [
        featureInputLayer(num_act, 'Normalization', 'none', 'Name', ['action_' suffix])
        fullyConnectedLayer(layer_sizes(1), 'Name', ['action_fc1_' suffix])
    ];

    lg = layerGraph(state_layers);
    lg = addLayers(lg, action_layers);
    lg = addLayers(lg, additionLayer(2, 'Name', ['add_' suffix]));
    lg = connectLayers(lg, ['state_relu1_' suffix], ['add_' suffix '/in1']);
    lg = connectLayers(lg, ['action_fc1_' suffix], ['add_' suffix '/in2']);

    previous = ['add_' suffix];
    for i = 2:numel(layer_sizes)
        fc = sprintf('critic_fc%d_%s', i, suffix);
        relu = sprintf('critic_relu%d_%s', i, suffix);
        lg = addLayers(lg, fullyConnectedLayer(layer_sizes(i), 'Name', fc));
        lg = addLayers(lg, reluLayer('Name', relu));
        lg = connectLayers(lg, previous, fc);
        lg = connectLayers(lg, fc, relu);
        previous = relu;
    end

    lg = addLayers(lg, fullyConnectedLayer(1, 'Name', ['q_value_' suffix]));
    lg = connectLayers(lg, previous, ['q_value_' suffix]);

    opts = rlRepresentationOptions('LearnRate', 3e-4, 'GradientThreshold', 1, 'UseDevice', 'cpu');
    critic = instantiate_qvalue_representation(lg, obs_info, act_info, ['state_' suffix], ['action_' suffix], opts);
end

function results = summarize_training(stats, training_time, maxEpisodes)
    results = struct();
    if isfield(stats, 'EpisodeReward')
        results.episode_rewards = stats.EpisodeReward;
        results.total_episodes = numel(stats.EpisodeReward);
        results.best_reward = max(stats.EpisodeReward);
        results.final_reward = stats.EpisodeReward(end);
        if isfield(stats, 'AverageReward')
            results.average_reward = stats.AverageReward(end);
        else
            tail = max(1, results.total_episodes - 4);
            results.average_reward = mean(stats.EpisodeReward(tail:end));
        end
    else
        results.episode_rewards = [];
        results.total_episodes = maxEpisodes;
        results.best_reward = NaN;
        results.final_reward = NaN;
        results.average_reward = NaN;
    end
    results.training_time = training_time;
end

function verify_continuous_actions(agent, obs_info, act_info, stage)
    deterministic_supported = false;
    original_flag = [];
    try
        deterministic_supported = isprop(agent, 'UseDeterministicExploitationPolicy');
        if deterministic_supported
            original_flag = agent.UseDeterministicExploitationPolicy;
            agent.UseDeterministicExploitationPolicy = true;
        end
        samples = 16;
        actions = zeros(samples, 1);
        for i = 1:samples
            obs = rand(obs_info.Dimension) * 0.6 + 0.2;
            out = getAction(agent, {obs});
            actions(i) = out{1};
        end
        if deterministic_supported
            agent.UseDeterministicExploitationPolicy = original_flag;
        end

        fprintf('%s动作分布: 范围[%.1f, %.1f] W, 标准差 %.1f W\n', stage, min(actions), max(actions), std(actions));
        if any(actions < act_info.LowerLimit - 100) || any(actions > act_info.UpperLimit + 100)
            fprintf('  ⚠ 动作越界\n');
        else
            fprintf('  ✅ 动作范围合理\n');
        end
    catch ME
        if deterministic_supported && ~isempty(original_flag)
            try
                agent.UseDeterministicExploitationPolicy = original_flag;
            catch
            end
        end
        fprintf('  ⚠ %s动作验证失败: %s\n', stage, ME.message);
    end
end

function value = get_option(options, field, default_value)
    if isstruct(options) && isfield(options, field)
        value = options.(field);
    else
        value = default_value;
    end
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
        for i = 1:numel(call_attempts)
            try
                args = call_attempts{i};
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
        for i = 1:numel(call_attempts)
            try
                args = call_attempts{i};
                critic = rlRepresentation(lg, obs_info, act_info, args{:});
                return;
            catch
            end
        end
    end

    error('SAC:MissingQValueAPI', '当前MATLAB版本不支持创建Q-value表示，请升级强化学习工具箱。');
end
