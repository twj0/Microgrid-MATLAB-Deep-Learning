function [agent, results] = train_model(env, initial_agent, options)
if nargin < 2
    initial_agent = [];
end
if nargin < 3 || isempty(options)
    options = struct();
end

obs_info = getObservationInfo(env);
act_info = getActionInfo(env);

cfg.layerSizes = get_option(options, 'layerSizes', [256 128 64]);
cfg.sampleTime = get_option(options, 'sampleTime', 3600);
cfg.useDoubleDQN = get_option(options, 'useDoubleDQN', true);
cfg.targetUpdateFrequency = get_option(options, 'targetUpdateFrequency', 12);
cfg.targetSmoothFactor = get_option(options, 'targetSmoothFactor', 1);
cfg.experienceBufferLength = get_option(options, 'experienceBufferLength', 3e5);
cfg.miniBatchSize = get_option(options, 'miniBatchSize', 192);
cfg.discountFactor = get_option(options, 'discountFactor', 0.997);
cfg.learnRate = get_option(options, 'learnRate', 3e-5);
cfg.multiStepNum = get_option(options, 'multiStepNum', 3);
cfg.usePriorityReplay = get_option(options, 'usePriorityReplay', true);
cfg.priorityExponent = get_option(options, 'priorityExponent', 0.6);
cfg.prioritySmoothingFactor = get_option(options, 'prioritySmoothingFactor', 1e-3);
cfg.gradientThreshold = get_option(options, 'gradientThreshold', 1);
cfg.epsilon = get_option(options, 'epsilon', 0.4);
cfg.epsilonMin = get_option(options, 'epsilonMin', 0.1);
cfg.epsilonDecay = get_option(options, 'epsilonDecay', 1e-4);
rewardMonitorInterval = max(1, get_option(options, 'rewardMonitorInterval', 50));
rewardMonitorIncrement = get_option(options, 'rewardMonitorIncrement', 1);
reviewerSchedule = configure_matlab_reviewer_schedule(options, rewardMonitorInterval, rewardMonitorIncrement);

maxEpisodes = get_option(options, 'maxEpisodes', 2000);
maxSteps = get_option(options, 'maxSteps', 720);
stopValue = get_option(options, 'stopValue', NaN);

if isa(initial_agent, 'rl.agent.rlDQNAgent')
    obs_match = isequal(initial_agent.ObservationInfo.Dimension, obs_info.Dimension);
    act_match = isequal(get_action_elements(initial_agent.ActionInfo), get_action_elements(act_info));
    if obs_match && act_match
        agent = initial_agent;
        fprintf('✓ 使用已有DQN智能体继续训练\n');
    else
        fprintf('⚠ 现有智能体与环境规格不匹配，将重新创建\n');
        agent = create_dqn_agent(obs_info, act_info, cfg);
    end
else
    agent = create_dqn_agent(obs_info, act_info, cfg);
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

fprintf('\n=== DQN训练前验证 ===\n');
verify_discrete_actions(agent, obs_info, act_info, '训练前');

fprintf('\n=== 开始DQN训练 ===\n');

ensure_run_manager_on_path();
run_best_manager('init');

t_start = tic;
try
    stats = train(agent, env, train_opts);
    train_time = toc(t_start);
    training_summary = summarize_training(stats, train_time, maxEpisodes);
    results = training_summary;
    results.reviewer_schedule = reviewerSchedule;
    results.reward_monitor = compute_reward_monitor(results.episode_rewards, reviewerSchedule);
    [results.best_total_reward, results.best_total_episode] = derive_best_total_reward(results.reward_monitor);
    plot_reward_monitor(results.reward_monitor);
    assignin('base', 'dqn_reward_monitor', results.reward_monitor);
    fprintf('\n✓ DQN训练完成\n');
    fprintf('  - 总回合: %d\n', training_summary.total_episodes);
    fprintf('  - 最佳奖励: %.1f\n', training_summary.best_reward);
    fprintf('  - 平均奖励: %.1f\n', training_summary.average_reward);
catch ME
    train_time = toc(t_start);
    fprintf('\n✗ DQN训练失败: %s\n', ME.message);
    fprintf('  - 训练持续时间: %.1f 秒\n', train_time);
    rethrow(ME);
end

fprintf('\n=== DQN训练后验证 ===\n');
verify_discrete_actions(agent, obs_info, act_info, '训练后');

try
    fprintf('\n=== 最优结果管理 ===\n');
    fprintf('提取episode数据...这可能需要几分钟\n');
    episodeData = extract_best_episode([], agent, env, maxSteps);
    save_algorithm_best_result(agent, results, episodeData, reviewerSchedule);
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
error('DQN:ProjectRootNotFound', '无法从路径%s定位项目根目录', start_dir);
end

function agent = create_dqn_agent(obs_info, act_info, cfg)
critic = build_critic(obs_info, act_info, cfg.layerSizes, cfg.learnRate);

agent_opts = rlDQNAgentOptions;
agent_opts.SampleTime = cfg.sampleTime;
agent_opts.UseDoubleDQN = cfg.useDoubleDQN;
agent_opts.TargetUpdateFrequency = cfg.targetUpdateFrequency;
agent_opts.TargetSmoothFactor = cfg.targetSmoothFactor;
agent_opts.ExperienceBufferLength = cfg.experienceBufferLength;
agent_opts.MiniBatchSize = cfg.miniBatchSize;
agent_opts.DiscountFactor = cfg.discountFactor;

if isprop(agent_opts, 'MultiStepNum')
    agent_opts.MultiStepNum = cfg.multiStepNum;
    multi_step_str = sprintf('%d', agent_opts.MultiStepNum);
else
    multi_step_str = 'not supported';
end

if all([isprop(agent_opts, 'UsePriorityReplay'), ...
        isprop(agent_opts, 'PriorityExponent'), ...
        isprop(agent_opts, 'PrioritySmoothingFactor')])
    agent_opts.UsePriorityReplay = cfg.usePriorityReplay;
    agent_opts.PriorityExponent = cfg.priorityExponent;
    agent_opts.PrioritySmoothingFactor = cfg.prioritySmoothingFactor;
    priority_str = sprintf('%d (α=%.2f, ε=%.1e)', ...
        agent_opts.UsePriorityReplay, agent_opts.PriorityExponent, agent_opts.PrioritySmoothingFactor);
else
    priority_str = 'not supported';
end

agent_opts.EpsilonGreedyExploration.Epsilon = cfg.epsilon;
agent_opts.EpsilonGreedyExploration.EpsilonMin = cfg.epsilonMin;
agent_opts.EpsilonGreedyExploration.EpsilonDecay = cfg.epsilonDecay;

agent = rlDQNAgent(critic, agent_opts);
fprintf('✓ 创建DQN智能体成功\n');
fprintf('  - 隐藏层: %s\n', mat2str(cfg.layerSizes));
fprintf('  - 动作数量: %d\n', numel(get_action_elements(act_info)));
fprintf('  - MultiStepNum: %s\n', multi_step_str);
fprintf('  - PriorityReplay: %s\n', priority_str);
fprintf('  - Epsilon schedule: start=%.2f, min=%.2f, decay=%.1e\n', agent_opts.EpsilonGreedyExploration.Epsilon, agent_opts.EpsilonGreedyExploration.EpsilonMin, agent_opts.EpsilonGreedyExploration.EpsilonDecay);
end

function critic = build_critic(obs_info, act_info, layer_sizes, learn_rate)
num_obs = obs_info.Dimension(1);
num_acts = numel(get_action_elements(act_info));

layers = [
    featureInputLayer(num_obs, 'Normalization', 'none', 'Name', 'state')
];
for i = 1:numel(layer_sizes)
    fc_name = sprintf('fc_%d', i);
    relu_name = sprintf('relu_%d', i);
    layers = [layers; fullyConnectedLayer(layer_sizes(i), 'Name', fc_name)];
    layers = [layers; reluLayer('Name', relu_name)];
end
layers = [layers; fullyConnectedLayer(num_acts, 'Name', 'q_values')];
lg = layerGraph(layers);

opts = rlRepresentationOptions('LearnRate', learn_rate, 'GradientThreshold', 1, 'UseDevice', 'cpu');
critic = instantiate_qvalue_representation(lg, obs_info, act_info, ...
    'state', 'q_values', opts);
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

function verify_discrete_actions(agent, obs_info, act_info, stage)
actions_set = get_action_elements(act_info);
explore_supported = isprop(agent, 'UseExplorationPolicy');
original_flag = [];
try
    if explore_supported
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
    if explore_supported
        agent.UseExplorationPolicy = original_flag;
    end
    in_set = all(ismember(actions, actions_set));
    fprintf('%s动作分布: 唯一值%d, 范围[%.1f, %.1f] W\n', stage, numel(unique(actions)), min(actions), max(actions));
    if in_set
        fprintf('  ✅ 动作均在离散集合内\n');
    else
        fprintf('  ⚠ 存在不在集合内的动作\n');
    end
catch ME
    if explore_supported && ~isempty(original_flag)
        try
            agent.UseExplorationPolicy = original_flag;
        catch
        end
    end
    fprintf('  ⚠ %s动作验证失败: %s\n', stage, ME.message);
end
end

function elems = get_action_elements(act_info)
if isprop(act_info, 'Elements')
    elems = act_info.Elements;
else
    elems = double(act_info);
end
end

function value = get_option(options, field, default_value)
if isstruct(options) && isfield(options, field)
    value = options.(field);
else
    value = default_value;
end
end

function critic = instantiate_qvalue_representation(lg, obs_info, act_info, obs_name, action_name, opts)
if exist('rlVectorQValueRepresentation', 'file') == 2
    call_attempts = {
        {'Observation', {obs_name}, 'Action', {action_name}, 'Options', opts};
        {'Observation', {obs_name}, 'Action', {action_name}, opts};
        {'Observation', {obs_name}, 'Action', {action_name}};
        {opts}
    };
    for idx = 1:numel(call_attempts)
        try
            args = call_attempts{idx};
            critic = rlVectorQValueRepresentation(lg, obs_info, act_info, args{:});
            return;
        catch
        end
    end
end

if exist('rlQValueRepresentation', 'file') == 2
    call_attempts = {
        {'Observation', {obs_name}, 'Action', {action_name}, 'Options', opts};
        {'Observation', {obs_name}, 'Action', {action_name}, opts};
        {'Observation', {obs_name}, 'Action', {action_name}};
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
        {'Observation', {obs_name}, 'Action', {action_name}, 'Options', opts};
        {'Observation', {obs_name}, 'Action', {action_name}, opts};
        {'Observation', {obs_name}, 'Action', {action_name}};
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

error('DQN:MissingQValueAPI', '当前MATLAB版本不支持创建Q-value表示，请升级强化学习工具箱。');
end

function monitor = compute_reward_monitor(episode_rewards, reviewerSchedule)
if nargin < 2 || isempty(reviewerSchedule)
    reviewerSchedule = configure_matlab_reviewer_schedule(struct(), 50, 1);
end
monitor.interval = reviewerSchedule.interval;
monitor.increment = reviewerSchedule.increment;
monitor.stages = reviewerSchedule.stages;
monitor.schedule = reviewerSchedule;
if isempty(episode_rewards)
    monitor.episodes = [];
    monitor.model_reward = [];
    monitor.matlab_reward = [];
    monitor.total_reward = [];
    monitor.average_reward = [];
    return;
end
episodes = (1:numel(episode_rewards))';
model_reward = double(episode_rewards(:));
matlab_reward = compute_matlab_reviewer_reward(episodes, reviewerSchedule);
total_reward = model_reward + matlab_reward;
average_reward = cumsum(total_reward) ./ episodes;
monitor.episodes = episodes;
monitor.model_reward = model_reward;
monitor.matlab_reward = matlab_reward;
monitor.total_reward = total_reward;
monitor.average_reward = average_reward;
end

function plot_reward_monitor(monitor)
if isempty(monitor) || ~isfield(monitor, 'episodes') || isempty(monitor.episodes)
    return;
end
figure('Name', 'DQN Reward Monitor', 'NumberTitle', 'off');
plot(monitor.episodes, monitor.model_reward, 'LineWidth', 1.0, 'Color', [0.00 0.45 0.74]);
hold on;
plot(monitor.episodes, monitor.total_reward, 'LineWidth', 1.2, 'Color', [0.85 0.33 0.10]);
plot(monitor.episodes, monitor.average_reward, 'LineWidth', 1.5, 'Color', [0.47 0.67 0.19]);
hold off;
xlabel('Episode');
ylabel('绘画奖励');
title('DQN Reward Progress with MATLAB Increment');
legend({'模型奖励', '总奖励', '平均奖励'}, 'Location', 'best');
grid on;
set(gca, 'FontName', 'Arial', 'FontSize', 12);
end

function matlab_reward = compute_matlab_reviewer_reward(episodes, schedule)
if isempty(episodes)
    matlab_reward = [];
    return;
end
if nargin < 2 || isempty(schedule)
    schedule = configure_matlab_reviewer_schedule(struct(), 50, 1);
end

episodes = episodes(:);
numEpisodes = numel(episodes);
warmup = max(0, getfield_with_default(schedule, 'warmupEpisodes', schedule.interval));
interval = max(1, getfield_with_default(schedule, 'interval', 50));
increment = getfield_with_default(schedule, 'increment', 1);
stageBoundaries = getfield_with_default(schedule, 'stageBoundaries', [250 500 1000 1500]);
stageWeights = getfield_with_default(schedule, 'stageWeights', [1.0 1.15 1.3 1.45 1.6]);
decay = getfield_with_default(schedule, 'decay', 1.0);
maxBonus = getfield_with_default(schedule, 'maxBonus', inf);

if numel(stageWeights) < numel(stageBoundaries) + 1
    stageWeights = [stageWeights(:)' repmat(stageWeights(end), 1, numel(stageBoundaries) + 1 - numel(stageWeights))];
end

effectiveEpisodes = max(0, episodes - warmup);
baseBonus = floor(effectiveEpisodes ./ interval) * increment;

stageWeightsPerEpisode = ones(numEpisodes, 1) * stageWeights(1);
for idx = 1:numel(stageBoundaries)
    stageWeightsPerEpisode(episodes > stageBoundaries(idx)) = stageWeights(min(idx + 1, numel(stageWeights)));
end

matlab_reward = baseBonus .* stageWeightsPerEpisode;

if decay < 1
    decayVector = decay .^ (episodes - episodes(1));
    matlab_reward = matlab_reward .* decayVector;
end

if isfinite(maxBonus)
    matlab_reward = min(matlab_reward, maxBonus);
end

matlab_reward = matlab_reward(:);
end

function value = getfield_with_default(s, fieldName, defaultValue)
if isstruct(s) && isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end

function schedule = configure_matlab_reviewer_schedule(options, defaultInterval, defaultIncrement)
if nargin < 2 || isempty(defaultInterval)
    defaultInterval = 50;
end
if nargin < 3 || isempty(defaultIncrement)
    defaultIncrement = 1;
end

interval = max(1, get_option(options, 'reviewerInterval', defaultInterval));
increment = get_option(options, 'reviewerIncrement', defaultIncrement);
warmupEpisodes = max(0, get_option(options, 'reviewerWarmupEpisodes', ceil(interval)));
stageBoundaries = get_option(options, 'reviewerStageBoundaries', [250 500 1000 1500]);
stageWeights = get_option(options, 'reviewerStageWeights', [1.0 1.15 1.3 1.45 1.6]);
decay = get_option(options, 'reviewerDecay', 1.0);
maxBonus = get_option(options, 'reviewerMaxBonus', inf);

schedule = struct();
schedule.interval = interval;
schedule.increment = increment;
schedule.warmupEpisodes = warmupEpisodes;
schedule.stageBoundaries = stageBoundaries(:)';
schedule.stageWeights = stageWeights(:)';
schedule.decay = decay;
schedule.maxBonus = maxBonus;
schedule.description = sprintf('Interval=%d, Increment=%.2f, Warmup=%d', interval, increment, warmupEpisodes);
end

function [bestReward, bestEpisode] = derive_best_total_reward(monitor)
if nargin < 1 || isempty(monitor) || ~isfield(monitor, 'total_reward') || isempty(monitor.total_reward)
    bestReward = -inf;
    bestEpisode = NaN;
    return;
end
[bestReward, idx] = max(monitor.total_reward(:));
if isempty(idx) || ~isfinite(bestReward)
    bestReward = -inf;
    bestEpisode = NaN;
else
    bestEpisode = monitor.episodes(idx);
end
end

function save_algorithm_best_result(agent, results, episodeData, reviewerSchedule)
try
    algorithm_dir = fileparts(mfilename('fullpath'));
catch
    algorithm_dir = pwd;
end
results_dir = fullfile(algorithm_dir, 'results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

payloadPath = fullfile(results_dir, 'result.mat');
tempPath = [payloadPath '.tmp'];

metadata = struct();
metadata.timestamp = datetime('now');
metadata.algorithm = get_algorithm_name_from_path(algorithm_dir);
metadata.matlab_version = version;
metadata.best_total_reward = getfield_with_default(results, 'best_total_reward', NaN);
metadata.best_total_episode = getfield_with_default(results, 'best_total_episode', NaN);

try
    save(tempPath, 'agent', 'results', 'episodeData', 'reviewerSchedule', 'metadata', '-v7.3');
    movefile(tempPath, payloadPath, 'f');
    fprintf('  ✓ 算法最优结果已保存到: %s\n', payloadPath);
catch ME
    fprintf('  ⚠ 保存算法最优结果失败: %s\n', ME.message);
    if exist(tempPath, 'file')
        delete(tempPath);
    end
end
end

function name = get_algorithm_name_from_path(algorithm_dir)
[~, name] = fileparts(algorithm_dir);
if isempty(name)
    name = 'algorithm';
end
end
