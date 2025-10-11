%% DDPG训练函数
% =========================================================================
% 功能: 创建和训练DDPG智能体
% 输入: env - RL环境对象
%       agent - (可选) 初始DDPG智能体
% 输出: agent - 训练好的DDPG智能体
%       training_results - 训练统计信息
% =========================================================================

function [agent, training_results] = train_model(env, agent)
    if nargin < 2
        agent = [];
    end
    
    fprintf('\n--- 初始化DDPG智能体 ---\n');
    
    %% 1. 获取环境信息
    % =====================================================================
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);
    
    numObservations = obsInfo.Dimension(1);
    numActions = actInfo.Dimension(1);
    
    fprintf('观测维度: %d\n', numObservations);
    fprintf('动作维度: %d\n', numActions);
    
    useExistingAgent = isa(agent, 'rl.agent.Agent');
    if useExistingAgent
        obsMatch = isequal(agent.ObservationInfo.Dimension, obsInfo.Dimension);
        actMatch = isequal(agent.ActionInfo.Dimension, actInfo.Dimension);
        if ~(obsMatch && actMatch)
            fprintf('\n警告: 传入的智能体与当前环境维度不匹配，将重新创建。\n');
            useExistingAgent = false;
        end
    end
    
    if useExistingAgent
        fprintf('\n✓ 使用传入的DDPG智能体继续训练\n');
    else
        %% 2. 创建Critic网络 (Q网络)
        % =====================================================================
        fprintf('\n创建Critic网络...\n');
        
        % Critic网络输入: [observations; actions]
        criticLayerSizes = [128, 64];  % 两层隐藏层
        
        % 状态路径
        statePath = [
            featureInputLayer(numObservations, 'Normalization', 'none', 'Name', 'observation')
            fullyConnectedLayer(criticLayerSizes(1), 'Name', 'CriticStateFC1')
            reluLayer('Name', 'CriticStateRelu1')
        ];
        
        % 动作路径
        actionPath = [
            featureInputLayer(numActions, 'Normalization', 'none', 'Name', 'action')
            fullyConnectedLayer(criticLayerSizes(1), 'Name', 'CriticActionFC1')
        ];
        
        % 合并路径
        commonPath = [
            additionLayer(2, 'Name', 'add')
            reluLayer('Name', 'CriticCommonRelu1')
            fullyConnectedLayer(criticLayerSizes(2), 'Name', 'CriticFC2')
            reluLayer('Name', 'CriticRelu2')
            fullyConnectedLayer(1, 'Name', 'CriticOutput')
        ];
        
        % 组装Critic网络
        criticNetwork = layerGraph(statePath);
        criticNetwork = addLayers(criticNetwork, actionPath);
        criticNetwork = addLayers(criticNetwork, commonPath);
        criticNetwork = connectLayers(criticNetwork, 'CriticStateRelu1', 'add/in1');
        criticNetwork = connectLayers(criticNetwork, 'CriticActionFC1', 'add/in2');
        
        % 创建Critic表示
        criticOptions = rlRepresentationOptions(...
            'LearnRate', 1e-3, ...
            'GradientThreshold', 1, ...
            'UseDevice', 'cpu');
        
        critic = rlQValueRepresentation(criticNetwork, obsInfo, actInfo, ...
            'Observation', {'observation'}, ...
            'Action', {'action'}, ...
            criticOptions);
        
        fprintf('✓ Critic网络创建完成\n');
        fprintf('  - 隐藏层: [%d, %d]\n', criticLayerSizes(1), criticLayerSizes(2));
        fprintf('  - 学习率: %.0e\n', 1e-3);
        
        %% 3. 创建Actor网络 (策略网络)
        % =====================================================================
        fprintf('\n创建Actor网络...\n');
        
        actorLayerSizes = [128, 64];  % 两层隐藏层
        
        % Actor网络设计: 输出连续的功率信号[-10kW, +10kW]
        actorNetwork = [
            featureInputLayer(numObservations, 'Normalization', 'none', 'Name', 'observation')
            fullyConnectedLayer(actorLayerSizes(1), 'Name', 'ActorFC1')
            reluLayer('Name', 'ActorRelu1')
            fullyConnectedLayer(actorLayerSizes(2), 'Name', 'ActorFC2')
            reluLayer('Name', 'ActorRelu2')
            fullyConnectedLayer(numActions, 'Name', 'ActorFC3')
            tanhLayer('Name', 'ActorTanh')  % 输出[-1, 1]
        ];
        
        % 将tanh输出[-1,1]缩放到动作空间[-UpperLimit, +UpperLimit]
        actorNetwork = [
            actorNetwork
            scalingLayer('Name', 'ActorScaling', 'Scale', actInfo.UpperLimit)
        ];
        
        % 创建Actor表示
        actorOptions = rlRepresentationOptions(...
            'LearnRate', 5e-4, ...
            'GradientThreshold', 1, ...
            'UseDevice', 'cpu');
        
        actor = rlDeterministicActorRepresentation(actorNetwork, obsInfo, actInfo, ...
            'Observation', {'observation'}, ...
            actorOptions);
        
        fprintf('✓ Actor网络创建完成\n');
        fprintf('  - 隐藏层: [%d, %d]\n', actorLayerSizes(1), actorLayerSizes(2));
        fprintf('  - 学习率: %.0e\n', 5e-4);
        fprintf('  - 动作范围: [%.1f, %.1f] kW\n', ...
            actInfo.LowerLimit/1000, actInfo.UpperLimit/1000);
        
        %% 4. 创建DDPG Agent
        % =====================================================================
        fprintf('\n创建DDPG Agent...\n');
        
        agentOptions = rlDDPGAgentOptions(...
            'SampleTime', 3600, ...                    % 1小时采样
            'TargetSmoothFactor', 1e-3, ...            % 目标网络软更新系数
            'DiscountFactor', 0.99, ...                % 折扣因子
            'MiniBatchSize', 64, ...                   % 批量大小
            'ExperienceBufferLength', 1e6);            % 经验回放缓冲区
        
        % 设置探索噪声 (使用兼容的方式)
        noise_span = (actInfo.UpperLimit - actInfo.LowerLimit) / 2;
        base_noise = 0.2 * noise_span;
        agentOptions.NoiseOptions.Mean = 0;
        agentOptions.NoiseOptions.MeanAttractionConstant = 1e-4;  % 适配3600s采样，确保稳定
        agentOptions.NoiseOptions.Variance = base_noise^2;
        agentOptions.NoiseOptions.VarianceDecayRate = 5e-5;
        
        agent = rlDDPGAgent(actor, critic, agentOptions);
        
        fprintf('✓ DDPG Agent创建完成\n');
        fprintf('  - 采样时间: %d 秒 (1小时)\n', 3600);
        fprintf('  - 折扣因子: %.2f\n', 0.99);
        fprintf('  - 批量大小: %d\n', 64);
        fprintf('  - 经验缓冲区: %.0e\n', 1e6);
    end
    
    %% 5. 配置训练选项
    % =====================================================================
    fprintf('\n配置训练参数...\n');
    
    trainOpts = rlTrainingOptions(...
        'MaxEpisodes', 500, ...                     % 最大训练回合数
        'MaxStepsPerEpisode', 720, ...             % 每回合最大步数 (30天*24小时)
        'ScoreAveragingWindowLength', 5, ...       % 平均窗口长度
        'Verbose', true, ...                       % 显示训练信息
        'Plots', 'training-progress', ...          % 显示训练曲线
        'StopTrainingCriteria', 'AverageReward', ...
        'StopTrainingValue', 500, ...              % 目标奖励值
        'SaveAgentCriteria', 'EpisodeReward', ...
        'SaveAgentValue', 300);                    % 保存阈值
    
    fprintf('✓ 训练参数配置完成\n');
    fprintf('  - 最大回合数: %d\n', 500);
    fprintf('  - 每回合步数: %d (30天)\n', 720);
    fprintf('  - 目标奖励: %.0f\n', 500);
    
    %% 6. 开始训练
    % =====================================================================
    fprintf('\n========================================\n');
    fprintf('  开始训练...\n');
    fprintf('========================================\n');
    
    tic;
    trainingStats = train(agent, env, trainOpts);
    training_time = toc;
    
    %% 7. 收集训练结果
    % =====================================================================
    training_results = struct();
    training_results.total_episodes = trainingStats.EpisodeIndex(end);
    training_results.episode_rewards = trainingStats.EpisodeReward;
    training_results.episode_steps = trainingStats.EpisodeSteps;
    training_results.training_time = training_time;
    training_results.average_reward = trainingStats.AverageReward(end);
    
    fprintf('\n========================================\n');
    fprintf('  训练统计\n');
    fprintf('========================================\n');
    fprintf('总回合数: %d\n', training_results.total_episodes);
    fprintf('平均奖励: %.2f\n', training_results.average_reward);
    fprintf('训练时长: %.2f 分钟\n', training_time/60);
    fprintf('========================================\n');
    
end
