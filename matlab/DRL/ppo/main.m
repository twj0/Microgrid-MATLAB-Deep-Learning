function main(options)
    if nargin < 1 || isempty(options)
        options = struct();
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

    clc;
    close all;

    fprintf('========================================\n');
    fprintf('  Microgrid PPO Reinforcement Learning Training v1.0\n');
    fprintf('========================================\n');
    fprintf('Start time: %s\n', char(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss")));

    try
        config = initialize_environment(options);
        fprintf('✓ Environment paths ready\n');

        try
            load_fuzzy_logic_system(config.model_dir);
            fprintf('✓ Fuzzy logic system loaded\n');
        catch ME
            fprintf('⚠ Failed to load fuzzy logic system: %s\n', ME.message);
        end

        [data, raw_data] = load_and_validate_data(config);
        fprintf('  - PV data: %.0f hours (%d samples)\n', data.duration_hours, data.sample_count);
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

        [agent, results] = train_model(env, initial_agent, config.training);
        fprintf('✓ PPO training completed\n');

        if exist('results', 'var') && ~isempty(results)
            assignin('base', 'ppo_training_results', results);
        end

        save_results(agent, results, config, agent_var_name);
        fprintf('✓ Training artifacts saved\n');

        verify_policy(agent, env);
        fprintf('✓ Policy verification finished\n');

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
        if exist('config', 'var')
            try
                if exist('results', 'var') && ~isempty(results)
                    assignin('base', 'ppo_training_results', results);
                    run_visualization(config, results);
                else
                    run_visualization(config, []);
                end
            catch vizErr
                fprintf('⚠ Visualization failed: %s\n', vizErr.message);
            end
        end
        rethrow(ME);
    end

    fprintf('========================================\n');
    fprintf('  Training pipeline finished\n');
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
        fprintf('⚠ visualization.m not found, skipping\n');
        return;
    end

    results_root = fullfile(project_root, 'results');
    if ~exist(results_root, 'dir')
        mkdir(results_root);
    end

    timestamp = char(datetime("now", "Format", "yyyyMMdd_HHmmss"));
    ppo_folder = fullfile(results_root, ['PPO_' timestamp]);
    if ~exist(ppo_folder, 'dir')
        mkdir(ppo_folder);
    end

    viz_options = struct( ...
        'workspace', "base", ...
        'saveFigures', true, ...
        'showFigures', false, ...
        'outputDir', ppo_folder, ...
        'filePrefix', "PPO", ...
        'figureFormat', "png", ...
        'closeAfterSave', true, ...
        'timestamp', timestamp ...
    );

    if ~isempty(trainingResults)
        viz_options.trainingResults = trainingResults;
    end

    visualization(viz_options);
    fprintf('✓ Visualization stored at: %s\n', ppo_folder);
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

    if ~exist(config.model_path, 'file')
        error('Simulink model not found: %s', config.model_path);
    end
    if ~exist(config.data_path, 'file')
        error('Data file not found: %s\nRun matlab/src/generate_data.m first.', config.data_path);
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

function [data, raw_data] = load_and_validate_data(config)
    raw_data = load(config.data_path);

    required_vars = {'pv_power_profile', 'load_power_profile'};
    for k = 1:numel(required_vars)
        if ~isfield(raw_data, required_vars{k})
            error('Missing required data: %s', required_vars{k});
        end
    end

    if isfield(raw_data, 'price_profile')
        price_profile = raw_data.price_profile;
    elseif isfield(raw_data, 'price_data')
        price_profile = raw_data.price_data;
    else
        error('Missing price data: price_profile or price_data');
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
        deficit = required_samples - data.sample_count;
        max_auto_pad = 24;
        if deficit <= max_auto_pad
            fprintf('⚠ Data deficit detected: %d samples (%.0f hours), padding with last value.\n', deficit, deficit * (config.sample_time/3600));

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
            data.duration_hours = calculate_duration_hours(data.pv_power);
        else
            error('Insufficient samples: need >=%d (%.0f hours), got %d (%.0f hours). Run matlab/src/generate_data.m.', required_samples, required_samples * (config.sample_time/3600), data.sample_count, data.duration_hours);
        end
    end
end

function [env, agent_var_name] = setup_simulink_environment(config)
    if ~bdIsLoaded(config.model_name)
        load_system(config.model_path);
    end
    set_param(config.model_name, 'StopTime', num2str(config.simulation_time));

    obs_info = rlNumericSpec([12 1]);
    obs_info.Name = 'Microgrid Observations';
    obs_info.Description = 'Battery_SOH, Battery_SOC, P_pv, P_load, Price, P_net_load, P_batt, P_grid, fuzzy_reward, economic_reward, health_reward, time_of_day';

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
        fprintf('  - Workspace variable updated: %s\n', agent_var_name);
    end

    timestamp = char(datetime("now", "Format", "yyyyMMdd_HHmmss"));
    filename = sprintf('ppo_agent_%s.mat', timestamp);
    metadata = struct('config', config, 'timestamp', datetime("now"), 'matlab_version', version);
    save(filename, 'agent', 'results', 'metadata');
    fprintf('  - Saved: %s\n', filename);
end

function verify_policy(agent, env)
    exploration_supported = false;
    old_flag = [];
    try
        obs_info = getObservationInfo(env);
        act_info = getActionInfo(env);
        exploration_supported = isprop(agent, 'UseExplorationPolicy');
        if exploration_supported
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

        if exploration_supported
            agent.UseExplorationPolicy = old_flag;
        end

        fprintf('  - Action range: [%.2f, %.2f] kW\n', min(actions)/1000, max(actions)/1000);
        fprintf('  - Action std: %.2f W\n', std(actions));

        if all(actions >= act_info.LowerLimit - 100) && all(actions <= act_info.UpperLimit + 100)
            fprintf('  ✅ Actions within limits\n');
        else
            fprintf('  ⚠ Action out of bounds\n');
        end
    catch ME
        if exploration_supported && ~isempty(old_flag)
            try
                agent.UseExplorationPolicy = old_flag;
            catch
            end
        end
        fprintf('  ⚠ Policy verification failed: %s\n', ME.message);
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
        fprintf('  - Created PPO placeholder agent: %s\n', agent_var_name);
    catch ME
        fprintf('  - Failed to create PPO placeholder agent: %s\n', ME.message);
    end
end

function agent = create_placeholder_agent(obs_info, act_info, sample_time)
    cfg.actorSizes = [64 32];
    cfg.criticSizes = [128 64];
    cfg.sampleTime = sample_time;
    cfg.stdScale = min((act_info.UpperLimit - act_info.LowerLimit) / 2 * 0.05, 500);
    agent = build_ppo_agent(obs_info, act_info, cfg);
end

function agent = build_ppo_agent(obs_info, act_info, cfg)
    actor = build_actor(obs_info, act_info, cfg.actorSizes, cfg.stdScale);
    critic = build_value(obs_info, cfg.criticSizes);

    agent_opts = rlPPOAgentOptions();
    agent_opts.SampleTime = cfg.sampleTime;
    agent_opts.ExperienceHorizon = 512;
    agent_opts.MiniBatchSize = 128;
    agent_opts.NumEpoch = 10;
    agent_opts.DiscountFactor = 0.99;
    agent_opts.GAEFactor = 0.95;
    agent_opts.AdvantageEstimateMethod = "gae";
    agent_opts.EntropyLossWeight = 0.01;

    call_attempts = {
        {actor, critic, agent_opts};
        {actor, critic};
        {struct('Actor', actor, 'Critic', critic, 'AgentOptions', agent_opts)}
    };

    agent = [];
    last_error = [];
    for idx = 1:numel(call_attempts)
        args = call_attempts{idx};
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
end

function actor = build_actor(obs_info, act_info, layer_sizes, std_scale)
    num_obs = obs_info.Dimension(1);
    num_act = act_info.Dimension(1);

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

    softplus_layer = softplusLayer('Name', 'action_std_softplus');
    scale_layer = scalingLayer('Name', 'action_std_scale', 'Scale', std_scale, 'Bias', 0.1);
    lg = addLayers(lg, softplus_layer);
    lg = addLayers(lg, scale_layer);
    lg = connectLayers(lg, 'action_logstd_fc', 'action_std_softplus');
    lg = connectLayers(lg, 'action_std_softplus', 'action_std_scale');

    actor_opts = rlRepresentationOptions('LearnRate', 3e-4, 'GradientThreshold', 1, 'UseDevice', 'cpu');
    actor = instantiate_gaussian_actor(lg, obs_info, act_info, 'action_mean_fc', 'action_std_scale', actor_opts);
end

function critic = build_value(obs_info, layer_sizes)
    num_obs = obs_info.Dimension(1);

    lg = layerGraph();
    lg = addLayers(lg, featureInputLayer(num_obs, 'Normalization', 'none', 'Name', 'state'));
    prev = 'state';
    for i = 1:numel(layer_sizes)
        fc_name = sprintf('value_fc%d', i);
        relu_name = sprintf('value_relu%d', i);
        lg = addLayers(lg, fullyConnectedLayer(layer_sizes(i), 'Name', fc_name));
        lg = addLayers(lg, reluLayer('Name', relu_name));
        lg = connectLayers(lg, prev, fc_name);
        lg = connectLayers(lg, fc_name, relu_name);
        prev = relu_name;
    end

    lg = addLayers(lg, fullyConnectedLayer(1, 'Name', 'value_output'));
    lg = connectLayers(lg, prev, 'value_output');

    critic_opts = rlRepresentationOptions('LearnRate', 3e-4, 'GradientThreshold', 1, 'UseDevice', 'cpu');
    critic = instantiate_value_representation(lg, obs_info, critic_opts);
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
        for idx = 1:numel(call_attempts)
            try
                args = call_attempts{idx};
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
        for idx = 1:numel(call_attempts)
            try
                args = call_attempts{idx};
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
