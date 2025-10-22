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
    % 修正：增大目标熵，减少探索噪声
    % 原始值-1太小，导致算法倾向于增加探索
    cfg.targetEntropy = get_option(options, 'targetEntropy', -1.0 * act_info.Dimension(1)); % 提高探索能力，避免策略过早收敛为确定性

    maxEpisodes = get_option(options, 'maxEpisodes', 2000);
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
    if isnan(stopValue) || ~isfinite(stopValue)
        train_opts.StopTrainingCriteria = 'EpisodeCount';
        train_opts.StopTrainingValue = maxEpisodes;
    else
        train_opts.StopTrainingCriteria = 'AverageReward';
        train_opts.StopTrainingValue = stopValue;
    end

    fprintf('\n=== SAC训练前验证 ===\n');
    verify_continuous_actions(agent, obs_info, act_info, '训练前');

    fprintf('\n=== 开始SAC训练 ===\n');

    % 初始化最优结果管理
    ensure_run_manager_on_path();
    run_best_manager('init');

    t_start = tic;
    % 在训练开始前初始化全局回合计数器（确保 Simulink 模型可以读取）
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
            training_summary = summarize_training(stats, train_time, maxEpisodes);
            results = training_summary;  % 将结果赋值给results变量
        end
        fprintf('\n✓ SAC训练完成\n');
        fprintf('  - 总回合: %d\n', results.total_episodes);
        fprintf('  - 最佳奖励: %.1f\n', results.best_reward);
        fprintf('  - 平均奖励: %.1f\n', results.average_reward);
    catch ME
        train_time = toc(t_start);
        fprintf('\n✗ SAC训练失败: %s\n', ME.message);
        fprintf('  - 训练持续时间: %.1f 秒\n', train_time);
        rethrow(ME);
    end

    fprintf('\n=== SAC训练后验证 ===\n');
    verify_continuous_actions(agent, obs_info, act_info, '训练后');

    % 提取并保存最优结果
    try
        fprintf('\n=== 最优结果管理 ===\n');
        fprintf('提取episode数据...这可能需要几分钟\n');
        episodeData = extract_best_episode([], agent, env, maxSteps);

        fprintf('\n比较并保存结果...\n');
        run_best_manager('save', agent, results, episodeData);
    catch ME
        fprintf('⚠ 最优结果保存失败: %s\n', ME.message);
        fprintf('  训练结果仍然有效,可继续使用\n');
    end
end

function ensure_run_manager_on_path()
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
error('SAC:ProjectRootNotFound', '无法从路径%s定位项目根目录', start_dir);
end

function agent = create_sac_agent(obs_info, act_info, cfg)
    actor = build_actor(obs_info, act_info, cfg.actorLayerSizes);
    critic1 = build_critic(obs_info, act_info, cfg.criticLayerSizes, 'c1');
    critic2 = build_critic(obs_info, act_info, cfg.criticLayerSizes, 'c2');

    agent_opts = rlSACAgentOptions();
    agent_opts.SampleTime = cfg.sampleTime;
    agent_opts.TargetSmoothFactor = 3e-3;  % 降低目标平滑系数Tau，目标网络更新更稳
    agent_opts.DiscountFactor = 0.985;  % 略降Gamma，减轻远期高幅度奖励对价值发散的影响
    agent_opts.ExperienceBufferLength = 1e6;
    agent_opts.MiniBatchSize = 384;  % 略增Batch以降低梯度方差（显存允许时）
    agent_opts.EntropyWeightOptions.TargetEntropy = cfg.targetEntropy;
    agent_opts.EntropyWeightOptions.LearnRate = 3e-4;  % 降低Alpha学习率，避免熵权震荡

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
    lg = addLayers(lg, fullyConnectedLayer(num_act, 'Name', 'actor_std_fc'));
    lg = addLayers(lg, softplusLayer('Name', 'actor_std_softplus'));
    % 修正：减小标准差缩放因子，避免动作总是达到边界值
    std_scale_factor = min(action_scale * 0.05, 500);  % 使用5%的动作范围或500的较小值
    lg = addLayers(lg, scalingLayer('Name', 'actor_std_scale', 'Scale', std_scale_factor, 'Bias', 0.1));

    lg = connectLayers(lg, previous, 'actor_mean_fc');
    lg = connectLayers(lg, previous, 'actor_std_fc');
    lg = connectLayers(lg, 'actor_std_fc', 'actor_std_softplus');
    lg = connectLayers(lg, 'actor_std_softplus', 'actor_std_scale');

    % 修正：降低学习率，提高训练稳定性
    opts = rlRepresentationOptions('LearnRate', 5e-5, 'GradientThreshold', 1, 'UseDevice', 'gpu');  % ActorLR
    mean_name = 'actor_mean_fc';
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

    opts = rlRepresentationOptions('LearnRate', 2e-4, 'GradientThreshold', 1, 'UseDevice', 'gpu');  % CriticLR
    critic = instantiate_qvalue_representation(lg, obs_info, act_info, ['state_' suffix], ['action_' suffix], opts);
end

function results = summarize_training(stats, training_time, maxEpisodes)
    results = struct();
    if isfield(stats, 'EpisodeReward') && ~isempty(stats.EpisodeReward)
        rewards = stats.EpisodeReward;
        if iscell(rewards)
            rewards = cellfun(@(x) convert_scalar(x), rewards);
        end
        rewards = double(rewards(:));

        results.episode_rewards = rewards;
        results.requested_episodes = maxEpisodes;
        results.recorded_episode_slots = numel(rewards);

        valid_reward_mask = ~isnan(rewards);
        results.total_episodes = nnz(valid_reward_mask);
        results.nan_episode_count = results.recorded_episode_slots - results.total_episodes;

        if results.total_episodes > 0
            results.best_reward = max(rewards(valid_reward_mask));
            last_valid_idx = find(valid_reward_mask, 1, 'last');
            results.final_reward = rewards(last_valid_idx);
            results.last_episode_index = last_valid_idx;
        else
            results.best_reward = NaN;
            results.final_reward = NaN;
            results.last_episode_index = NaN;
        end

        if isfield(stats, 'AverageReward') && ~isempty(stats.AverageReward)
            avg_values = stats.AverageReward;
            if iscell(avg_values)
                avg_values = cellfun(@(x) convert_scalar(x), avg_values);
            end
            avg_values = double(avg_values(:));
            valid_avg_mask = ~isnan(avg_values);
            if any(valid_avg_mask)
                last_avg_idx = find(valid_avg_mask, 1, 'last');
                results.average_reward = avg_values(last_avg_idx);
            else
                results.average_reward = NaN;
            end
        else
            tail_start = max(1, results.total_episodes - 4);
            tail_rewards = rewards(valid_reward_mask);
            if results.total_episodes >= 1
                effective_tail = tail_rewards(max(1, end - (results.total_episodes - tail_start)):end);
                results.average_reward = mean(effective_tail);
            else
                results.average_reward = NaN;
            end
        end
    else
        results.episode_rewards = [];
        results.total_episodes = maxEpisodes;
        results.best_reward = NaN;
        results.final_reward = NaN;
        results.average_reward = NaN;
        results.requested_episodes = maxEpisodes;
        results.recorded_episode_slots = 0;
        results.nan_episode_count = maxEpisodes;
        results.last_episode_index = NaN;
    end
    results.training_time = training_time;
end

function value = convert_scalar(x)
    if isempty(x)
        value = NaN;
    else
        value = double(x);
    end
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

function [agent, results] = train_with_online_best_saving(agent, env, base_opts, maxEpisodes, maxSteps)
% 	6	6	6
%
%
% 逐	5 9 	 	 	 	 	 	 	 	 
%
    bestReward = -inf;
    allRewards = [];
    t_start_local = tic;
    % 初始化全局回合计数器，供奖励函数读取（向后兼容：无影响）
    assignin('base','g_episode_num',0);

    for ep = 1:maxEpisodes
        % 每回合开始前写入当前回合号到 base workspace
        assignin('base','g_episode_num', ep);
        % 统一的 Episode Reset 日志输出（不改变外部接口）
        fprintf('[ResetFcn] Episode %d reset at %s\n', ep, char(datetime('now','Format','HH:mm:ss')));
        opts = base_opts;
        try
            opts.StopTrainingCriteria = 'EpisodeCount';
        catch
            % R2025a
        end
        opts.StopTrainingValue = 1;

        stats = train(agent, env, opts);
        if isfield(stats, 'EpisodeReward')
            r = double(stats.EpisodeReward(end));
        else
            r = NaN;
        end
        allRewards(end+1) = r; %#ok<AGROW>

        if r > bestReward
            bestReward = r;
            try
                %
                episodeData = extract_best_episode([], agent, env, maxSteps);
                tmp = struct();
                tmp.episode_rewards = allRewards;
                tmp.best_reward = bestReward;
                tmp.total_episodes = numel(allRewards);
                tmp.average_reward = mean(allRewards);
                tmp.training_time = toc(t_start_local);
                run_best_manager('save', agent, tmp, episodeData);
            catch ME
                fprintf('	 	: %s\n', ME.message);
            end
        end
    end

    results = struct();
    results.episode_rewards = allRewards;
    results.best_reward = bestReward;
    results.total_episodes = numel(allRewards);
    results.average_reward = mean(allRewards);
    results.training_time = toc(t_start_local);
end


