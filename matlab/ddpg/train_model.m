%% 统一DDPG训练函数 - 集成所有训练策略
% =========================================================================
% 功能: 整合train_ddpg_agent.m, train_ddpg_simple.m, train_model.m
% 特色: 根据模式参数选择不同的训练策略
% 版本: 统一版 v3.0
% =========================================================================

function [agent, training_results] = train_model(env, initial_agent, mode)
    % 输入参数:
    %   env - RL环境对象
    %   initial_agent - 可选，初始智能体
    %   mode - 训练模式
    %     'advanced' - 高级训练 (修正版，最佳性能)
    %     'simple'   - 兼容训练 (最大兼容性)
    %     'legacy'   - 原始训练 (保持原有逻辑)
    
    if nargin < 2
        initial_agent = [];
    end
    if nargin < 3
        mode = 'advanced';
    end
    
    fprintf('\n=== 统一DDPG训练 ===\n');
    fprintf('训练模式: %s\n', upper(mode));
    
    % 获取环境信息
    obs_info = getObservationInfo(env);
    act_info = getActionInfo(env);
    
    num_obs = obs_info.Dimension(1);
    num_act = act_info.Dimension(1);
    
    fprintf('观测维度: %d\n', num_obs);
    fprintf('动作维度: %d\n', num_act);
    fprintf('动作范围: [%.1f, %.1f] kW\n', act_info.LowerLimit/1000, act_info.UpperLimit/1000);
    
    % 检查是否使用现有智能体
    use_existing_agent = isa(initial_agent, 'rl.agent.Agent');
    if use_existing_agent
        obs_match = isequal(initial_agent.ObservationInfo.Dimension, obs_info.Dimension);
        act_match = isequal(initial_agent.ActionInfo.Dimension, act_info.Dimension);
        if ~(obs_match && act_match)
            fprintf('⚠ 现有智能体维度不匹配，将重新创建\n');
            use_existing_agent = false;
        else
            fprintf('✓ 使用现有智能体继续训练\n');
        end
    end
    
    if use_existing_agent
        agent = initial_agent;
    else
        % 根据模式创建智能体
        switch lower(mode)
            case 'advanced'
                agent = create_advanced_agent(obs_info, act_info);
            case 'simple'
                agent = create_simple_agent(obs_info, act_info);
            case 'legacy'
                agent = create_legacy_agent(obs_info, act_info);
            otherwise
                error('未知训练模式: %s', mode);
        end
    end
    
    % 根据模式配置训练选项
    train_opts = configure_training_options(mode);
    
    % 训练前验证连续动作输出
    fprintf('\n=== 训练前验证 ===\n');
    verify_continuous_action_output(agent, obs_info, act_info, '训练前');
    
    % 开始训练
    fprintf('\n=== 开始训练 ===\n');
    tic;
    try
        training_stats = train(agent, env, train_opts);
        training_time = toc;
        
        % 整理训练结果
        training_results = process_training_results(training_stats, training_time, mode);
        
        fprintf('\n✓ 训练完成!\n');
        display_training_summary(training_results, mode);
        
        % 训练后验证
        fprintf('\n=== 训练后验证 ===\n');
        verify_continuous_action_output(agent, obs_info, act_info, '训练后');
        
    catch ME
        fprintf('\n✗ 训练失败: %s\n', ME.message);
        training_duration = toc;
        training_results = struct('error', ME.message, 'total_episodes', 0, 'training_time', training_duration);
        fprintf('  - 错误上下文: %s\n', training_results.error);
        rethrow(ME);
    end
end

%% 创建高级智能体 (修正版，最佳性能)
function agent = create_advanced_agent(obs_info, act_info)
    fprintf('\n创建高级DDPG智能体...\n');
    
    num_obs = obs_info.Dimension(1);
    num_act = act_info.Dimension(1);
    
    %% 1. 创建Critic网络（状态-动作融合架构）
    % 状态处理分支
    state_layers = [
        featureInputLayer(num_obs, 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(256, 'Name', 'state_fc1')
        reluLayer('Name', 'state_relu1')
        fullyConnectedLayer(128, 'Name', 'state_fc2')
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
        fullyConnectedLayer(128, 'Name', 'common_fc1')
        reluLayer('Name', 'common_relu1')
        fullyConnectedLayer(64, 'Name', 'common_fc2')
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
    action_range = act_info.UpperLimit - act_info.LowerLimit;
    action_scale = action_range / 2;  % tanh输出[-1,1]，需要缩放到范围的一半
    action_bias = (act_info.UpperLimit + act_info.LowerLimit) / 2;  % 中心点偏移
    
    actor_layers = [
        featureInputLayer(num_obs, 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(256, 'Name', 'actor_fc1')
        reluLayer('Name', 'actor_relu1')
        dropoutLayer(0.1, 'Name', 'dropout1')
        fullyConnectedLayer(128, 'Name', 'actor_fc2')
        reluLayer('Name', 'actor_relu2')
        dropoutLayer(0.1, 'Name', 'dropout2')
        fullyConnectedLayer(64, 'Name', 'actor_fc3')
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
        'SampleTime', 3600, ...
        'TargetSmoothFactor', 1e-3, ...
        'DiscountFactor', 0.99, ...
        'MiniBatchSize', 64, ...
        'ExperienceBufferLength', 1e6);
    
    % 版本自适应噪声配置
    try
        noise_std = action_scale * 0.3;
        if exist('rlOrnsteinUhlenbeckActionNoise', 'file')
            noise_options = rlOrnsteinUhlenbeckActionNoise(...
                'Mean', 0, ...
                'StandardDeviation', noise_std, ...
                'MeanAttractionConstant', 0.15, ...
                'StandardDeviationDecayRate', 1e-5);
            agent_options.NoiseOptions = noise_options;
            fprintf('  ✓ 使用新版本噪声配置\n');
        else
            agent_options.NoiseOptions.Variance = noise_std^2;
            agent_options.NoiseOptions.VarianceDecayRate = 1e-5;
            fprintf('  ✓ 使用兼容版本噪声配置\n');
        end
    catch
        fprintf('  - 使用默认噪声配置\n');
    end
    
    agent = rlDDPGAgent(actor, critic, agent_options);
    
    fprintf('✓ 高级DDPG智能体创建完成\n');
    fprintf('  - Actor网络: [256→128→64] + Scaling(%.1f, %.1f)\n', action_scale, action_bias);
    fprintf('  - Critic网络: 状态-动作融合架构\n');
end

%% 创建简化智能体 (兼容版，最大兼容性)
function agent = create_simple_agent(obs_info, act_info)
    fprintf('\n创建简化DDPG智能体...\n');
    
    num_obs = obs_info.Dimension(1);
    num_act = act_info.Dimension(1);
    
    % 计算缩放参数
    action_range = act_info.UpperLimit - act_info.LowerLimit;
    action_scale = action_range / 2;
    action_bias = (act_info.UpperLimit + act_info.LowerLimit) / 2;
    
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
    agent_options.SampleTime = 3600;
    agent_options.DiscountFactor = 0.99;
    agent_options.MiniBatchSize = 32;
    agent_options.ExperienceBufferLength = 50000;
    
    try
        agent_options.TargetSmoothFactor = 1e-3;
    catch
        % 忽略不支持的参数
    end
    
    agent = rlDDPGAgent(actor, critic, agent_options);
    
    fprintf('✓ 简化DDPG智能体创建完成\n');
    fprintf('  - Actor网络: [128→64] + Scaling(%.1f, %.1f)\n', action_scale, action_bias);
    fprintf('  - Critic网络: 简化融合架构\n');
end

%% 创建原始智能体 (保持原有逻辑)
function agent = create_legacy_agent(obs_info, act_info)
    fprintf('\n创建原始DDPG智能体...\n');
    
    num_obs = obs_info.Dimension(1);
    num_act = act_info.Dimension(1);
    
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
        'SampleTime', 3600, ...
        'TargetSmoothFactor', 1e-3, ...
        'DiscountFactor', 0.99, ...
        'MiniBatchSize', 64, ...
        'ExperienceBufferLength', 1e6);
    
    % 原始噪声配置
    noise_span = (act_info.UpperLimit - act_info.LowerLimit) / 2;
    base_noise = 0.2 * noise_span;
    agent_options.NoiseOptions.Mean = 0;
    agent_options.NoiseOptions.MeanAttractionConstant = 1e-4;
    agent_options.NoiseOptions.Variance = base_noise^2;
    agent_options.NoiseOptions.VarianceDecayRate = 5e-5;
    
    agent = rlDDPGAgent(actor, critic, agent_options);
    
    fprintf('✓ 原始DDPG智能体创建完成\n');
    fprintf('  - 注意: 使用原始配置，可能存在已知问题\n');
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
