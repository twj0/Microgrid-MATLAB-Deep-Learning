

function main(options)
if nargin < 1 || isempty(options)
    options = struct();
end

clc;
close all;

fprintf('========================================\n');
fprintf('  微电网DQN强化学习训练系统 v1.0\n');
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
    fprintf('  - 负载数据: %.0f 小时 (%d 样本)\n', calculate_duration_hours(data.load_power), calculate_sample_count(data.load_power));
    fprintf('  - 电价数据: %.0f 小时 (%d 样本)\n', calculate_duration_hours(data.price_profile), calculate_sample_count(data.price_profile));
    push_simulation_data_to_base(data, raw_data);
    fprintf('✓ 数据加载与分发完成\n');

    [env, agent_var_name, action_set] = setup_simulink_environment(config);
    fprintf('✓ Simulink环境配置完成\n');

    initial_agent = [];
    if ~isempty(agent_var_name)
        try
            initial_agent = evalin('base', agent_var_name);
        catch
            initial_agent = [];
        end
    end

    config.training.actionSet = action_set;
    [agent, results] = train_model(env, initial_agent, config.training);
    fprintf('✓ DQN训练完成\n');

    if exist('results', 'var') && ~isempty(results)
        assignin('base', 'dqn_training_results', results);
    end

    save_results(agent, results, config, action_set, agent_var_name);
    fprintf('✓ 训练结果已保存\n');

    verify_policy(agent, action_set);
    fprintf('✓ 策略验证完成\n');

    try
        run_visualization(config, results);
    catch vizErr
        fprintf('⚠ 可视化生成失败: %s\n', vizErr.message);
    end

catch ME
    fprintf('\n✗ 发生错误: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('位置: %s (第%d行)\n', ME.stack(1).name, ME.stack(1).line);
    end
    if exist('config', 'var')
        try
            if exist('results', 'var') && ~isempty(results)
                assignin('base', 'dqn_training_results', results);
                run_visualization(config, results);
            else
                run_visualization(config, []);
            end
        catch vizErr
            fprintf('⚠ 可视化生成失败: %s\n', vizErr.message);
        end
    end
    rethrow(ME);
end

fprintf('========================================\n');
fprintf('  训练流程结束\n');
fprintf('========================================\n');
end

function run_visualization(config, trainingResults)
if nargin < 2
    trainingResults = [];
end

project_root = fileparts(config.model_dir);
matlab_src_dir = fullfile(project_root, 'matlab', 'src');
if exist(matlab_src_dir, 'dir')
    addpath(matlab_src_dir);
end
expected_visualization = fullfile(matlab_src_dir, 'visualization.m');
visualization_path = which('visualization');
if isempty(visualization_path) || ~strcmpi(visualization_path, expected_visualization)
    fprintf('⚠ 未找到visualization函数，跳过可视化生成\n');
    return;
end

results_root = fullfile(project_root, 'results');
if ~exist(results_root, 'dir')
    mkdir(results_root);
end

timestamp = char(datetime("now", "Format", "yyyyMMdd_HHmmss"));
dqn_folder = fullfile(results_root, ['DQN_' timestamp]);
if ~exist(dqn_folder, 'dir')
    mkdir(dqn_folder);
end

viz_options = struct(...
    'workspace', "base", ...
    'saveFigures', true, ...
    'showFigures', false, ...
    'outputDir', dqn_folder, ...
    'filePrefix', "DQN", ...
    'figureFormat', "png", ...
    'closeAfterSave', true, ...
    'timestamp', timestamp ...
);

if ~isempty(trainingResults)
    viz_options.trainingResults = trainingResults;
end

visualization(viz_options);
fprintf('✓ 可视化已保存至: %s\n', dqn_folder);
end

function config = initialize_environment(options)
script_dir = fileparts(mfilename('fullpath'));
project_root = find_project_root(script_dir);
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

config.training.maxEpisodes = get_option(options, 'maxEpisodes', 2000);
config.training.maxSteps = get_option(options, 'maxSteps', 720);
config.training.stopValue = get_option(options, 'stopValue', Inf);
config.training.sampleTime = config.sample_time;
config.training.layerSizes = get_option(options, 'layerSizes', [256 128 64]);
config.training.useDoubleDQN = get_option(options, 'useDoubleDQN', true);
config.training.targetUpdateFrequency = get_option(options, 'targetUpdateFrequency', 4);
config.training.targetSmoothFactor = get_option(options, 'targetSmoothFactor', 1);
config.training.experienceBufferLength = get_option(options, 'experienceBufferLength', 1e6);
config.training.miniBatchSize = get_option(options, 'miniBatchSize', 256);
config.training.discountFactor = get_option(options, 'discountFactor', 0.99);
config.training.learnRate = get_option(options, 'learnRate', 1e-4);
config.training.multiStepNum = get_option(options, 'multiStepNum', 3);
config.training.usePriorityReplay = get_option(options, 'usePriorityReplay', true);
config.training.priorityExponent = get_option(options, 'priorityExponent', 0.6);
config.training.prioritySmoothingFactor = get_option(options, 'prioritySmoothingFactor', 1e-3);
config.training.gradientThreshold = get_option(options, 'gradientThreshold', 1);
config.training.epsilon = get_option(options, 'epsilon', 0.6);
config.training.epsilonMin = get_option(options, 'epsilonMin', 0.05);
config.training.epsilonDecay = get_option(options, 'epsilonDecay', 2e-4);
config.training.actionResolution = get_option(options, 'actionResolution', 1000);
config.training.actionLimit = get_option(options, 'actionLimit', 10e3);

if ~exist(config.model_path, 'file')
    error('未找到Simulink模型: %s', config.model_path);
end
if ~exist(config.data_path, 'file')
    error('未找到数据文件: %s\n请先运行 matlab/src/generate_data.m', config.data_path);
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

function [env, agent_var_name, action_set] = setup_simulink_environment(config)
if ~bdIsLoaded(config.model_name)
    load_system(config.model_path);
end
set_param(config.model_name, 'StopTime', num2str(config.simulation_time));

obs_info = rlNumericSpec([12 1]);
obs_info.Name = 'Microgrid Observations';
obs_info.Description = 'Battery_SOH, Battery_SOC, P_pv, P_load, Price, P_net_load, P_batt, P_grid, fuzzy_reward, economic_reward, health_reward, time_of_day';

action_limit = config.training.actionLimit;
action_resolution = max(1, config.training.actionResolution);
action_set = generate_action_set(action_limit, action_resolution);
act_info = rlFiniteSetSpec(action_set);
act_info.Name = 'Battery Power Set';

agent_block = [config.model_name '/RL Agent'];
agent_var_name = ensure_agent_variable(agent_block, obs_info, act_info, config.sample_time);

env = rlSimulinkEnv(config.model_name, agent_block, obs_info, act_info);
env.ResetFcn = @(in)setVariable(in, 'initial_soc', 50);
end

function save_results(agent, results, config, action_set, agent_var_name)
if ~isempty(agent_var_name)
    assignin('base', agent_var_name, agent);
    fprintf('  - 工作区变量已更新: %s\n', agent_var_name);
end

timestamp = char(datetime("now", "Format", "yyyyMMdd_HHmmss"));
filename = sprintf('dqn_agent_%s.mat', timestamp);
metadata = struct('config', config, 'timestamp', datetime("now"), 'matlab_version', version, 'action_set', action_set);
save(filename, 'agent', 'results', 'metadata');
fprintf('  - 已保存: %s\n', filename);
end

function verify_policy(agent, action_set)
try
    obs_info = agent.ObservationInfo;
    explore_supported = isprop(agent, 'UseExplorationPolicy');
    old_flag = [];
    if explore_supported
        old_flag = agent.UseExplorationPolicy;
        agent.UseExplorationPolicy = false;
    end

    samples = 20;
    actions = zeros(samples, 1);
    for i = 1:samples
        obs = rand(obs_info.Dimension) * 0.6 + 0.2;
        action_cell = getAction(agent, {obs});
        actions(i) = action_cell{1};
    end

    if explore_supported
        agent.UseExplorationPolicy = old_flag;
    end

    unique_count = numel(unique(actions));
    valid = all(ismember(actions, action_set));
    fprintf('  - 动作唯一值: %d\n', unique_count);
    fprintf('  - 动作范围: [%.1f, %.1f] kW\n', min(actions)/1000, max(actions)/1000);
    if valid
        fprintf('  ✅ 动作集合合法\n');
    else
        fprintf('  ⚠ 动作存在越界\n');
    end
catch ME
    if exist('explore_supported', 'var') && explore_supported && ~isempty(old_flag)
        try
            agent.UseExplorationPolicy = old_flag;
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
    if isa(existing, 'rl.agent.rlDQNAgent')
        return;
    end
catch
end

try
    placeholder = create_placeholder_agent(obs_info, act_info, sample_time);
    assignin('base', agent_var_name, placeholder);
    fprintf('  - 已创建DQN占位智能体: %s\n', agent_var_name);
catch ME
    fprintf('  - DQN占位智能体创建失败: %s\n', ME.message);
end
end

function agent = create_placeholder_agent(obs_info, act_info, sample_time)
cfg.layerSizes = [256 128 64];
cfg.learnRate = 1e-4;
cfg.sampleTime = sample_time;
cfg.useDoubleDQN = true;
cfg.targetUpdateFrequency = 4;
cfg.targetSmoothFactor = 1;
cfg.experienceBufferLength = 5e5;
cfg.miniBatchSize = 256;
cfg.discountFactor = 0.99;
cfg.multiStepNum = 3;
cfg.usePriorityReplay = true;
cfg.priorityExponent = 0.6;
cfg.prioritySmoothingFactor = 1e-3;
cfg.epsilonMin = 0.05;
cfg.gradientThreshold = 1;
cfg.epsilon = 0.6;
cfg.epsilonDecay = 2e-4;
agent = build_placeholder_agent(obs_info, act_info, cfg);
end

function agent = build_placeholder_agent(obs_info, act_info, cfg)
critic = build_placeholder_critic(obs_info, act_info, cfg.layerSizes, cfg.learnRate);
opts = rlDQNAgentOptions;
opts.SampleTime = cfg.sampleTime;
opts.UseDoubleDQN = cfg.useDoubleDQN;
opts.TargetUpdateFrequency = cfg.targetUpdateFrequency;
opts.TargetSmoothFactor = cfg.targetSmoothFactor;
opts.ExperienceBufferLength = cfg.experienceBufferLength;
opts.MiniBatchSize = cfg.miniBatchSize;
opts.DiscountFactor = cfg.discountFactor;
if isprop(opts, 'MultiStepNum')
    opts.MultiStepNum = cfg.multiStepNum;
end
if isprop(opts, 'UsePriorityReplay')
    opts.UsePriorityReplay = cfg.usePriorityReplay;
end
if isprop(opts, 'PriorityExponent')
    opts.PriorityExponent = cfg.priorityExponent;
end
if isprop(opts, 'PrioritySmoothingFactor')
    opts.PrioritySmoothingFactor = cfg.prioritySmoothingFactor;
end
opts.EpsilonGreedyExploration.Epsilon = cfg.epsilon;
opts.EpsilonGreedyExploration.EpsilonMin = cfg.epsilonMin;
opts.EpsilonGreedyExploration.EpsilonDecay = cfg.epsilonDecay;
agent = rlDQNAgent(critic, opts);
end
function critic = build_placeholder_critic(obs_info, act_info, layer_sizes, learn_rate)
num_obs = obs_info.Dimension(1);
num_acts = numel(act_info.Elements);

layers = [
    featureInputLayer(num_obs, 'Normalization', 'none', 'Name', 'state')
];
for i = 1:numel(layer_sizes)
    fc_name = sprintf('placeholder_fc_%d', i);
    relu_name = sprintf('placeholder_relu_%d', i);
    layers = [layers; fullyConnectedLayer(layer_sizes(i), 'Name', fc_name)];
    layers = [layers; reluLayer('Name', relu_name)];
end
layers = [layers; fullyConnectedLayer(num_acts, 'Name', 'placeholder_q_values')];
lg = layerGraph(layers);

opts = rlRepresentationOptions('LearnRate', learn_rate, 'GradientThreshold', 1, 'UseDevice', 'cpu');
critic = instantiate_qvalue_representation(lg, obs_info, act_info, ...
    'state', 'placeholder_q_values', opts);
end

function action_set = generate_action_set(action_limit, action_resolution)
if action_limit <= 0
    action_limit = 10e3;
end
step = max(1, action_resolution);
values = -action_limit:step:action_limit;
if ~ismember(0, values)
    values = sort([values 0]);
end
if numel(values) > 1
    midpoints = (values(1:end-1) + values(2:end)) / 2;
    values = [values midpoints];
end
action_set = sort(unique(values));
action_set = double(action_set(:))';
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
