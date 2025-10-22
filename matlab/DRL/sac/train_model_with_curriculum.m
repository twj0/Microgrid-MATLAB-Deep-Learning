function [agent, results] = train_model_with_curriculum(env, initial_agent, options)
%TRAIN_MODEL_WITH_CURRICULUM SAC训练 + 课程学习集成版本
%   在原有train_model基础上集成动态参数调整功能
%
% 输入:
%   env           - Simulink环境对象
%   initial_agent - 初始智能体(可选)
%   options       - 训练选项结构体
%
% 输出:
%   agent   - 训练后的智能体
%   results - 训练结果统计
%
% 选项参数:
%   options.maxEpisodes        - 最大训练回合数(默认: 2000)
%   options.maxSteps           - 每回合最大步数(默认: 720)
%   options.stopValue          - 停止训练的奖励阈值(默认: NaN)
%   options.curriculum_enabled - 是否启用课程学习(默认: true)
%   options.pause_interval     - 暂停间隔(默认: 50)
%   options.auto_adjust        - 是否自动调整(默认: true)
%   options.interactive        - 是否交互模式(默认: false)
%
% 示例:
%   % 1. 使用默认课程学习
%   [agent, results] = train_model_with_curriculum(env, [], struct());
%
%   % 2. 自定义课程学习参数
%   opts = struct('maxEpisodes', 1000, 'pause_interval', 100);
%   [agent, results] = train_model_with_curriculum(env, [], opts);
%
%   % 3. 禁用课程学习(回退到标准训练)
%   opts = struct('curriculum_enabled', false);
%   [agent, results] = train_model_with_curriculum(env, [], opts);

%% ========================================================================
% 1. 参数解析与初始化
%% ========================================================================

if nargin < 2
    initial_agent = [];
end

if nargin < 3 || isempty(options)
    options = struct();
end

% 提取训练参数
obs_info = getObservationInfo(env);
act_info = getActionInfo(env);

% Agent配置
cfg.actorLayerSizes = get_option(options, 'actorLayerSizes', [256 128 64]);
cfg.criticLayerSizes = get_option(options, 'criticLayerSizes', [256 128 64]);
cfg.sampleTime = get_option(options, 'sampleTime', 3600);
cfg.targetEntropy = get_option(options, 'targetEntropy', -0.5 * act_info.Dimension(1));

% 训练参数
maxEpisodes = get_option(options, 'maxEpisodes', 2000);
maxSteps = get_option(options, 'maxSteps', 720);
stopValue = get_option(options, 'stopValue', NaN);

% 课程学习参数
curriculum_enabled = get_option(options, 'curriculum_enabled', true);
pause_interval = get_option(options, 'pause_interval', 50);
auto_adjust = get_option(options, 'auto_adjust', true);
interactive_mode = get_option(options, 'interactive', false);

%% ========================================================================
% 2. 创建或使用现有Agent
%% ========================================================================

if isempty(initial_agent)
    fprintf('\n=== 创建SAC Agent ===\n');
    agent = create_sac_agent(obs_info, act_info, cfg);
else
    fprintf('\n=== 使用提供的Agent ===\n');
    agent = initial_agent;
end

%% ========================================================================
% 3. 配置训练选项
%% ========================================================================

fprintf('\n=== 配置训练选项 ===\n');

% 基础训练选项
train_opts = rlTrainingOptions;
train_opts.MaxEpisodes = maxEpisodes;
train_opts.MaxStepsPerEpisode = maxSteps;
train_opts.Plots = 'training-progress';
train_opts.ScoreAveragingWindowLength = max(1, min(5, maxEpisodes));
train_opts.Verbose = ~curriculum_enabled;  % 课程学习模式下关闭默认输出

if isnan(stopValue) || ~isfinite(stopValue)
    train_opts.StopTrainingCriteria = 'EpisodeCount';
    train_opts.StopTrainingValue = maxEpisodes;
else
    train_opts.StopTrainingCriteria = 'AverageReward';
    train_opts.StopTrainingValue = stopValue;
end

fprintf('  - 最大回合: %d\n', maxEpisodes);
fprintf('  - 每回合步数: %d\n', maxSteps);
fprintf('  - 课程学习: %s\n', bool_to_str(curriculum_enabled));

%% ========================================================================
% 4. 训练前验证
%% ========================================================================

fprintf('\n=== SAC训练前验证 ===\n');
verify_continuous_actions(agent, obs_info, act_info, '训练前');

%% ========================================================================
% 5. 执行训练
%% ========================================================================

fprintf('\n=== 开始SAC训练 ===\n');

% 初始化最优结果管理
ensure_run_manager_on_path();
run_best_manager('init');

t_start = tic;

try
    if curriculum_enabled
        % 使用课程学习训练
        fprintf('  模式: 课程学习 (动态参数调整)\n');
        fprintf('  暂停间隔: 每 %d 个episode\n', pause_interval);
        
        % 添加训练框架路径
        training_framework_path = fullfile(fileparts(mfilename('fullpath')), ...
            '..', '..', 'src', 'training');
        addpath(training_framework_path);
        
        % 配置课程学习
        curriculum_config = struct(...
            'pause_interval', pause_interval, ...
            'auto_adjust', auto_adjust, ...
            'interactive', interactive_mode ...
        );
        
        % 执行课程学习训练
        [agent, results] = train_with_dynamic_params(...
            agent, env, train_opts, curriculum_config);
        
    else
        % 使用标准训练
        fprintf('  模式: 标准训练 (无课程学习)\n');
        
        stats = train(agent, env, train_opts);
        train_time = toc(t_start);
        
        % 整理结果
        results = summarize_training(stats, train_time, maxEpisodes);
    end
    
    train_time = toc(t_start);
    
    fprintf('\n✓ SAC训练完成\n');
    fprintf('  - 总回合: %d\n', results.total_episodes);
    fprintf('  - 最佳奖励: %.1f\n', results.best_reward);
    fprintf('  - 平均奖励: %.1f\n', results.average_reward);
    fprintf('  - 训练时长: %.1f 分钟\n', train_time / 60);
    
catch ME
    train_time = toc(t_start);
    fprintf('\n✗ SAC训练失败: %s\n', ME.message);
    fprintf('  - 训练持续时间: %.1f 秒\n', train_time);
    rethrow(ME);
end

%% ========================================================================
% 6. 训练后验证
%% ========================================================================

fprintf('\n=== SAC训练后验证 ===\n');
verify_continuous_actions(agent, obs_info, act_info, '训练后');

%% ========================================================================
% 7. 提取并保存最优结果
%% ========================================================================

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

%% ========================================================================
% 辅助函数
%% ========================================================================

function agent = create_sac_agent(obs_info, act_info, cfg)
    %创建SAC agent
    
    % Actor网络
    actor_layers = [
        featureInputLayer(obs_info.Dimension(1))
        fullyConnectedLayer(cfg.actorLayerSizes(1))
        reluLayer
        fullyConnectedLayer(cfg.actorLayerSizes(2))
        reluLayer
        fullyConnectedLayer(cfg.actorLayerSizes(3))
        reluLayer
        fullyConnectedLayer(act_info.Dimension(1))
        tanhLayer
        scalingLayer('Scale', act_info.UpperLimit)
    ];
    
    actor = rlDeterministicActorRepresentation(actor_layers, obs_info, act_info);
    
    % Critic网络
    critic_layers = [
        featureInputLayer(obs_info.Dimension(1) + act_info.Dimension(1))
        fullyConnectedLayer(cfg.criticLayerSizes(1))
        reluLayer
        fullyConnectedLayer(cfg.criticLayerSizes(2))
        reluLayer
        fullyConnectedLayer(cfg.criticLayerSizes(3))
        reluLayer
        fullyConnectedLayer(1)
    ];
    
    critic = rlQValueRepresentation(critic_layers, obs_info, act_info);
    
    % Agent选项
    agent_opts = rlSACAgentOptions(...
        'SampleTime', cfg.sampleTime, ...
        'DiscountFactor', 0.985, ...
        'TargetSmoothFactor', 3e-3, ...
        'MiniBatchSize', 256, ...
        'ExperienceBufferLength', 5e5, ...
        'TargetEntropy', cfg.targetEntropy);
    
    % 创建agent
    agent = rlSACAgent(actor, critic, agent_opts);
    
    fprintf('  ✓ SAC Agent创建完成\n');
    fprintf('    - Actor网络: %s\n', mat2str(cfg.actorLayerSizes));
    fprintf('    - Critic网络: %s\n', mat2str(cfg.criticLayerSizes));
    fprintf('    - 目标熵: %.2f\n', cfg.targetEntropy);
end

function value = get_option(options, field_name, default_value)
    %安全获取选项值
    if isfield(options, field_name)
        value = options.(field_name);
    else
        value = default_value;
    end
end

function str = bool_to_str(bool_val)
    %布尔值转字符串
    if bool_val
        str = '启用';
    else
        str = '禁用';
    end
end

