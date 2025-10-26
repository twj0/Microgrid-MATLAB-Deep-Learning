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
    action_span = act_info.UpperLimit - act_info.LowerLimit;
    cfg.stdScale = get_option(options, 'stdScale', min(action_span / 2 * 0.10, 1000)); % 提高策略标准差初值，增强探索

    maxEpisodes = get_option(options, 'maxEpisodes', 2000);
    maxSteps = get_option(options, 'maxSteps', 720);
    stopValue = get_option(options, 'stopValue', NaN);

    if isa(initial_agent, 'rl.agent.rlPPOAgent')
        obs_match = isequal(initial_agent.ObservationInfo.Dimension, obs_info.Dimension);
        act_match = isequal(initial_agent.ActionInfo.Dimension, act_info.Dimension);
        if obs_match && act_match
            agent = initial_agent;
            fprintf('✓ Continue training with existing PPO agent\n');
        else
            fprintf('⚠ Existing agent incompatible with environment, creating new PPO agent\n');
            agent = create_ppo_agent(obs_info, act_info, cfg);
        end
    else
        agent = create_ppo_agent(obs_info, act_info, cfg);
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

    fprintf('\n=== PPO Pre-training validation ===\n');
    verify_continuous_actions(agent, obs_info, act_info, 'Pre-training');

    fprintf('\n=== PPO Training start ===\n');

    ensure_run_manager_on_path();
    run_best_manager('init');

    t_start = tic;
    % 在训练开始前初始化全局回合计数器（确保 Simulink 模型等可读取）
    if ~evalin('base', 'exist(''g_episode_num'', ''var'')')
        assignin('base', 'g_episode_num', 0);
    end

    try
        % 新增开关：逐回合训练+实时最优保存（默认 false，可由外部开启）
        saveBestFlag = get_option(options, 'saveBestDuringTraining', false);
        useAdaptiveEntropy = get_option(options, 'useAdaptiveEntropy', true); % 默认开启自适应熵，防止策略分布过早收缩
        entropyAdjustEvery = get_option(options, 'entropyAdjustEvery', 100); % 调整频率为100，平衡稳定与探索
        if saveBestFlag
            [agent, results] = train_with_online_best_saving(agent, env, train_opts, maxEpisodes, maxSteps, useAdaptiveEntropy, entropyAdjustEvery);
            train_time = results.training_time; % 由辅助函数内部统计
        else
            % Plan A: Cosine annealing via nested ResetFcn (single-train)
            baseReset = env.ResetFcn; if isempty(baseReset), baseReset = @(in) in; end
            lr_trace = [];
            cfgLR = struct('mode','cosine','Tmax',maxEpisodes, ...
                           'baseA',3e-4,'minA',3e-5, ...
                           'baseC',3e-4,'minC',3e-5);
            env.ResetFcn = @reset_with_lr;
            stats = train(agent, env, train_opts);
            train_time = toc(t_start);
            training_summary = summarize_training(stats, train_time, maxEpisodes);
            results = training_summary;
            try, results.lr_trace = lr_trace; catch, end
        end
        fprintf('\n✓ PPO training completed\n');
        fprintf('  - Episodes: %d\n', results.total_episodes);
        fprintf('  - Best reward: %.1f\n', results.best_reward);
        fprintf('  - Average reward: %.1f\n', results.average_reward);
    catch ME
        train_time = toc(t_start);
        fprintf('\n✗ PPO training failed: %s\n', ME.message);
        fprintf('  - Training duration: %.1f s\n', train_time);
        rethrow(ME);
    end

    fprintf('\n=== PPO Post-training validation ===\n');
    verify_continuous_actions(agent, obs_info, act_info, 'Post-training');

    try
        fprintf('\n=== Best run extraction ===\n');
        fprintf('Collecting episode data... may take a while\n');
        episodeData = extract_best_episode([], agent, env, maxSteps);

        fprintf('\nComparing with stored best run...\n');
        run_best_manager('save', agent, results, episodeData);
    catch ME
        fprintf('⚠ Saving best run failed: %s\n', ME.message);
        fprintf('  Training results remain usable\n');
    end
    % --- Nested ResetFcn for LR cosine annealing (Plan A) ---
    function in = reset_with_lr(in)
        % Maintain global episode counter for Simulink/RL blocks
        ep = 1;
        try
            ep = evalin('base','g_episode_num') + 1;
        catch
            ep = 1;
        end
        assignin('base','g_episode_num', ep);

        % Compute and apply learning rates
        lr = compute_lr(ep, cfgLR);
        actorLR = lr.actor; criticLR = lr.critic;
        agent = drl_set_lr(agent, actorLR, criticLR);

        % Record learning rate trace: [episode, actorLR, criticLR]
        lr_trace(end+1, :) = [double(ep) double(actorLR) double(criticLR)]; %#ok<AGROW>

        % Call original ResetFcn
        in = baseReset(in);
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
error('PPO:ProjectRootNotFound', 'Unable to locate project root from %s', start_dir);
end

function agent = create_ppo_agent(obs_info, act_info, cfg)
    actor = build_actor(obs_info, act_info, cfg.actorLayerSizes, cfg.stdScale);
    critic = build_value(obs_info, cfg.criticLayerSizes);

    agent_opts = rlPPOAgentOptions();
    agent_opts.SampleTime = cfg.sampleTime;
    agent_opts.ExperienceHorizon = 512;
    agent_opts.MiniBatchSize = 128;
    agent_opts.NumEpoch = 10;
    agent_opts.DiscountFactor = 0.99;
    agent_opts.GAEFactor = 0.95;
    agent_opts.AdvantageEstimateMethod = "gae";
    agent_opts.EntropyLossWeight = 0.02; % 略增熵权重，增强探索避免策略塌缩

    call_attempts = {
        {actor, critic, agent_opts};
        {actor, critic};
        {struct('Actor', actor, 'Critic', critic, 'AgentOptions', agent_opts)}
    };

    agent = [];
    last_error = [];
    for i = 1:numel(call_attempts)
        args = call_attempts{i};
        try
            agent = rlPPOAgent(args{:});
            last_error = [];
            break;
        catch ME
            last_error = ME;
        end
    end

    if isempty(agent)
        rethrow(last_error);
    end

    fprintf('✓ PPO agent created\n');
    fprintf('  - Actor layers: %s\n', mat2str(cfg.actorLayerSizes));
    fprintf('  - Critic layers: %s\n', mat2str(cfg.criticLayerSizes));
end

function actor = build_actor(obs_info, act_info, layer_sizes, std_scale)
    num_obs = obs_info.Dimension(1);
    num_act = act_info.Dimension(1);

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
    lg = addLayers(lg, scalingLayer('Name', 'actor_std_scale', 'Scale', std_scale, 'Bias', 0.1));

    lg = connectLayers(lg, previous, 'actor_mean_fc');
    lg = connectLayers(lg, previous, 'actor_std_fc');
    lg = connectLayers(lg, 'actor_std_fc', 'actor_std_softplus');
    lg = connectLayers(lg, 'actor_std_softplus', 'actor_std_scale');

    opts = rlRepresentationOptions('LearnRate', 3e-4, 'GradientThreshold', 1, 'UseDevice', 'gpu');
    actor = instantiate_gaussian_actor(lg, obs_info, act_info, 'actor_mean_fc', 'actor_std_scale', opts);
end

function critic = build_value(obs_info, layer_sizes)
    num_obs = obs_info.Dimension(1);

    lg = layerGraph();
    lg = addLayers(lg, featureInputLayer(num_obs, 'Normalization', 'none', 'Name', 'state'));
    previous = 'state';
    for i = 1:numel(layer_sizes)
        fc = sprintf('value_fc_%d', i);
        relu = sprintf('value_relu_%d', i);
        lg = addLayers(lg, fullyConnectedLayer(layer_sizes(i), 'Name', fc));
        lg = addLayers(lg, reluLayer('Name', relu));
        lg = connectLayers(lg, previous, fc);
        lg = connectLayers(lg, fc, relu);
        previous = relu;
    end

    lg = addLayers(lg, fullyConnectedLayer(1, 'Name', 'value_output'));
    lg = connectLayers(lg, previous, 'value_output');

    opts = rlRepresentationOptions('LearnRate', 3e-4, 'GradientThreshold', 1, 'UseDevice', 'gpu');
    critic = instantiate_value_representation(lg, obs_info, opts);
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
        deterministic_supported = isprop(agent, 'UseExplorationPolicy');
        if deterministic_supported
            original_flag = agent.UseExplorationPolicy;
            agent.UseExplorationPolicy = false;
        end
        samples = 16;
        actions = zeros(samples, 1);
        for i = 1:samples
            obs = rand(obs_info.Dimension) * 0.6 + 0.2;
            out = getAction(agent, {obs});
            actions(i) = out{1};
        end
        if deterministic_supported
            agent.UseExplorationPolicy = original_flag;
        end

        fprintf('%s action distribution: range[%.1f, %.1f] W, std %.1f W\n', stage, min(actions), max(actions), std(actions));
        if any(actions < act_info.LowerLimit - 100) || any(actions > act_info.UpperLimit + 100)
            fprintf('  ⚠ Action out of bounds\n');
        else
            fprintf('  ✅ Action range valid\n');
        end
    catch ME
        if deterministic_supported && ~isempty(original_flag)
            try
                agent.UseExplorationPolicy = original_flag;
            catch
            end
        end
        fprintf('  ⚠ %s action validation failed: %s\n', stage, ME.message);
    end
end

function actor = instantiate_gaussian_actor(lg, obs_info, act_info, mean_name, std_name, opts)
    if exist('rlContinuousGaussianActorRepresentation', 'file') == 2
        actor = rlContinuousGaussianActorRepresentation(lg, obs_info, act_info, 'Observation', {'state'}, 'Action', {mean_name}, 'ActionStd', {std_name}, 'Options', opts);
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
        error('PPO:StochasticActorCreationFailed', 'Unable to create stochastic actor with known signatures.');
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

    error('PPO:MissingGaussianActorAPI', 'Current MATLAB version does not support Gaussian actor creation.');
end

function critic = instantiate_value_representation(lg, obs_info, opts)
    if exist('rlValueRepresentation', 'file') == 2
        call_attempts = {
            {'Observation', {'state'}, 'Options', opts};
            {'Observation', {'state'}, opts};
            {'Observation', {'state'}};
            {opts}
        };
        for i = 1:numel(call_attempts)
            try
                args = call_attempts{i};
                critic = rlValueRepresentation(lg, obs_info, args{:});
                return;
            catch
            end
        end
    end

    if exist('rlRepresentation', 'file') == 2
        call_attempts = {
            {'Observation', {'state'}, 'Options', opts};
            {'Observation', {'state'}, opts};
            {'Observation', {'state'}};
            {opts}
        };
        for i = 1:numel(call_attempts)
            try
                args = call_attempts{i};
                critic = rlRepresentation(lg, obs_info, args{:});
                return;
            catch
            end
        end
    end

    error('PPO:MissingValueAPI', 'Current MATLAB version does not support value representation creation.');
end

function value = get_option(options, field, default_value)
    if isstruct(options) && isfield(options, field)
        value = options.(field);
    else
        value = default_value;
    end
end

function [agent, results] = train_with_online_best_saving(agent, env, base_opts, maxEpisodes, maxSteps, useAdaptiveEntropy, entropyAdjustEvery)
    % 逐回合训练并在回合奖励超过历史最优时实时保存（仅保留一个最优模型）
    if nargin < 6, useAdaptiveEntropy = true; end  % 默认自适应熵开启
    if nargin < 7, entropyAdjustEvery = 100; end % 默认每100回合调整一次
    minEntropyW = 1e-4; maxEntropyW = 5e-2; % 熵权重上下限

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
        end
        opts.StopTrainingValue = 1;
        opts.Plots = 'training-progress';

        stats = train(agent, env, opts);
        if isfield(stats, 'EpisodeReward')
            r = double(stats.EpisodeReward(end));
        else
            r = NaN;
        end
        allRewards(end+1) = r; %#ok<AGROW>

        %  
        if useAdaptiveEntropy && ep >= entropyAdjustEvery && mod(ep, entropyAdjustEvery) == 0
            try
                recent = allRewards(max(1, end-entropyAdjustEvery+1):end);
                prev   = allRewards(max(1, end-2*entropyAdjustEvery+1):max(1, end-entropyAdjustEvery));
                if ~isempty(prev)
                    w = agent.AgentOptions.EntropyLossWeight;
                    if mean(recent) > mean(prev) + 1e-6
                        w = max(minEntropyW, 0.9*w); % 
                    else
                        w = min(maxEntropyW, 1.1*w); % 
                    end
                    agent.AgentOptions.EntropyLossWeight = w;



                end
            catch
                % 
            end
        end

        if r > bestReward
            bestReward = r;
            try
                episodeData = extract_best_episode([], agent, env, maxSteps);
                tmp = struct();
                tmp.episode_rewards = allRewards;
                tmp.best_reward = bestReward;
                tmp.total_episodes = numel(allRewards);
                tmp.average_reward = mean(allRewards);
                tmp.training_time = toc(t_start_local);
                run_best_manager('save', agent, tmp, episodeData);
            catch ME
                fprintf('⚠ PPO online save failed: %s\n', ME.message);
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


function agent = drl_set_lr(agent, actorLR, criticLR)
% Safely update Actor/Critic learn rates; skip parts with NaN
try
    if isfinite(actorLR)
        a = getActor(agent);
        if ~isempty(a)
            if numel(a) > 1
                for i = 1:numel(a), a(i).Options.LearnRate = actorLR; end
            else
                a.Options.LearnRate = actorLR;
            end
            agent = setActor(agent, a);
        end
    end
catch
end
try
    if isfinite(criticLR)
        C = getCritic(agent);
        if ~isempty(C)
            if numel(C) > 1
                for k = 1:numel(C), C(k).Options.LearnRate = criticLR; end
            else
                C.Options.LearnRate = criticLR;
            end
            agent = setCritic(agent, C);
        end
    end
catch
end
% SAC: also align entropy weight optimizer LR to the smaller one
try
    v = min([actorLR, criticLR]); if isfinite(v), agent.AgentOptions.EntropyWeightOptions.LearnRate = v; end
catch
end
end

function lr = compute_lr(ep, cfg)
% Cosine annealing from base -> min across [1..Tmax]
Tmax = max(1, cfg.Tmax);
t = min((ep-1)/max(1, Tmax-1), 1);
s = 0.5*(1 + cos(pi*t));
lr.actor  = cfg.minA + (cfg.baseA - cfg.minA) * s;
lr.critic = cfg.minC + (cfg.baseC - cfg.minC) * s;
end
