function env = create_env(simulation_days)
%CREATE_ENV 构建并返回 rlSimulinkEnv 以用于重放或诊断
%   用途：为 test_data_extraction.m 或其他脚本提供标准 env 构建入口。
%   - 模型: model/Microgrid.slx
%   - Agent 块: 'Microgrid/RL Agent'
%   - 观测规格: rlNumericSpec([12 1])
%   - 动作规格: rlNumericSpec([1 1])，范围[-10kW, +10kW]
%
%   调用示例：
%       env = create_env(30);  % 30天仿真时间（默认）
%
%   返回：
%       env (rlSimulinkEnv)

    if nargin < 1 || isempty(simulation_days)
        simulation_days = 30; % 与训练脚本一致的缺省值
    end

    % 定位项目根目录与模型路径
    thisFile = mfilename('fullpath');
    runMgrDir = fileparts(thisFile);
    project_root = find_project_root(runMgrDir);
    model_name = 'Microgrid';
    model_path = fullfile(project_root, 'model', [model_name '.slx']);

    % 将必要路径加入搜索路径
    addpath(genpath(fullfile(project_root, 'matlab', 'src')));
    addpath(fullfile(project_root, 'model'));

    % 加载模型并设置仿真时间
    if ~bdIsLoaded(model_name)
        load_system(model_path);
    end
    sample_time = 3600; % 秒
    stop_time = simulation_days * 24 * sample_time;
    set_param(model_name, 'StopTime', num2str(stop_time));

    % 定义观测/动作规格（与DRL各算法保持一致）
    obs_info = rlNumericSpec([12 1]);
    obs_info.Name = 'Microgrid Observations';

    act_info = rlNumericSpec([1 1]);
    act_info.LowerLimit = -10e3; % W
    act_info.UpperLimit =  10e3; % W
    act_info.Name = 'Battery Power';

    % Agent 块路径（需在模型中存在名为“RL Agent”的块）
    agent_block = [model_name '/RL Agent'];

    % 构建环境（显式关闭 Fast Restart，避免回放时工作区/日志异常）
    env = rlSimulinkEnv(model_name, agent_block, obs_info, act_info, 'UseFastRestart', false);

    % 可选：保持默认 ResetFcn（不设置），避免依赖外部函数
    % env.ResetFcn = @(in) in; % 如需在每回合设置初始变量，可自定义此函数

    % 最后，返回 env
end

function project_root = find_project_root(start_dir)
%FIND_PROJECT_ROOT 自下而上查找同时包含 matlab/ 与 model/ 的目录
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
    error('create_env:ProjectRootNotFound', '无法从路径%s定位项目根目录', start_dir);
end

