function test_data_extraction(maxSteps)
%TEST_DATA_EXTRACTION 诊断提取最优 episode 的时序数据（关闭 Fast Restart 并打印 logsout）
%   用途：训练后如 best_episode_data.mat 为空或缺少电池数据，运行本脚本重放一次仿真并打印诊断信息。
%
% 调用示例：
%   test_data_extraction(720)
%
if nargin < 1
    maxSteps = 720; % 例如 30 天 * 24 = 720
end

fprintf('\n=== 测试：最优 episode 数据提取诊断 ===\n');

% 确保路径
thisFile = mfilename('fullpath');
runMgrDir = fileparts(thisFile);
repoRoot = find_project_root(runMgrDir);
addpath(genpath(fullfile(repoRoot,'matlab','src')));
addpath(fullfile(repoRoot,'model'));

% 加载最优 agent 与既有 episode 数据（如有）
% 路径已添加，确认 run_best_manager 可用
if ~exist('run_best_manager','file')
    error('run_best_manager 未在路径上，请确认已执行 addpath(genpath(''matlab/src''))');
end
[agent, episodeDataLoaded, meta] = run_best_manager('load');
if isempty(agent)
    error('未找到 best_agent.mat，请先完成训练。');
end

% 获取/构造 env（优先使用 base workspace 中已有的 env）
try
    if evalin('base','exist(''env'',''var'')')
        env = evalin('base','env');
        fprintf('✓ 使用工作区中的 env\n');
    elseif isfield(meta,'env_builder') && isa(meta.env_builder,'function_handle')
        env = meta.env_builder();
        fprintf('✓ 使用 meta.env_builder 构建 env\n');
    else
        fprintf('⚠ 未在 workspace 或 meta 中找到 env，尝试调用 create_env()/build_env() ...\n');
        env = [];
        try, env = create_env(); fprintf('✓ 使用 create_env() 构建 env\n'); end
        if isempty(env)
            try, env = build_env(); fprintf('✓ 使用 build_env() 构建 env\n'); end
        end
    end
catch ME
    fprintf('⚠ 构建 env 失败: %s\n', ME.message);
    env = [];
end

if isempty(env)
    fprintf('\n✗ 无法获得 env。请在工作区提供变量 env（rlSimulinkEnv），或在 meta 中提供 env_builder。\n');
    return;
end

% 推断模型名 & 关闭 Fast Restart，开启 Signal Logging
modelName = '';
try
    if isprop(env,'Model')
        mdl = env.Model;
        if isstring(mdl), mdl = char(mdl); end
        if ischar(mdl) && ~isempty(mdl)
            modelName = mdl;
        end
    end
    if isempty(modelName) && isprop(env,'ModelName')
        mdl = env.ModelName;
        if isstring(mdl), mdl = char(mdl); end
        if ischar(mdl) && ~isempty(mdl)
            modelName = mdl;
        end
    end
    if isempty(modelName)
        try, modelName = env.mdl; catch, end
    end
catch
end

if ~isempty(modelName)
    try
        if ~bdIsLoaded(modelName), load_system(modelName); end
    catch
    end
    try, set_param(modelName,'FastRestart','off');                catch, end
    try, set_param(modelName,'SignalLogging','on');               catch, end
    try, set_param(modelName,'SignalLoggingName','logsout');      catch, end
    try, set_param(modelName,'SignalLoggingSaveFormat','Dataset');catch, end
    fprintf('DEBUG: 已关闭 FastRestart 并启用 SignalLogging -> logsout（模型：%s）\n', modelName);
else
    fprintf('DEBUG: 未能推断模型名，可能无法设置 FastRestart/SignalLogging\n');
end

% 打印模型 Signal Logging 配置参数
if ~isempty(modelName)
    try
        siglog = get_param(modelName, 'SignalLogging');
        siglogName = get_param(modelName, 'SignalLoggingName');
        siglogFmt  = get_param(modelName, 'SignalLoggingSaveFormat');
        fprintf('Model SignalLogging: %s, Name: %s, Format: %s\n', siglog, siglogName, siglogFmt);
    catch ME
        fprintf('读取模型日志配置失败: %s\n', ME.message);
    end
end
    %         
    try
        logInfo = get_param(modelName, 'SignalLoggingInfo');
        fprintf('SignalLoggingInfo \n');
        disp(logInfo);
    catch
    end

    %   
    try
        load_system(modelName);
        ln = find_system(modelName, 'FindAll','on','Type','line');
        names = {};
        for k=1:numel(ln)
            nm = '';
            try, nm = string(get_param(ln(k),'Name')); catch, end
            if strlength(nm)>0
                names{end+1} = nm; %#ok<AGROW>
            end
        end
        if ~isempty(names)
            fprintf('Model  (%d):\n', numel(names));
            % : soc/soh/batt/cost
            patt = @(s) contains(lower(s),'soc') || contains(lower(s),'soh') || contains(lower(s),'batt') || contains(lower(s),'cost');
            sel = names(cellfun(@(s) patt(char(s)), names));
            if isempty(sel)
                fprintf('     SOC/SOH/BATT/COST \n');
            else
                fprintf('   SOC/SOH/BATT/COST :\n');
                for k=1:numel(sel)
                    fprintf('    - %s\n', sel{k});
                end
            end
        else
            fprintf('Model  \n');
        end
    catch ME
        fprintf('  : %s\n', ME.message);
    end



    try
        print_struct_fields(simOut, 'simOut', 2);
    catch
    end


% 执行一次重放仿真（直接调用 Simulink sim，确保 To Workspace/Signal Logging 可用）
try
    % 将 agent 放入 base workspace，确保 RL Agent 块引用到
    try, assignin('base','agent',agent); catch, end

    % 建议：批量标记目标信号为 Log（如用户已在模型中勾选，此步骤无副作用）
    try
        mark_signals_for_logging(modelName, {'Battery_SOC','Battery_SOH','P_batt','Battery_Power','TotalCost','SOH_Diff','SOC','SOH'});
    catch
    end

    % 关闭 FastRestart；尽量返回单一 SimulationOutput（部分版本默认已开启）
    try, set_param(modelName,'FastRestart','off'); end

    % 直接仿真模型（通过 RL Agent 块使用 agent），返回 SimulationOutput
    simOut = sim(modelName, 'CaptureErrors','on');
catch ME
    fprintf('\n✗ 直接 sim(modelName) 失败: %s\n', ME.message);
    return;
end


% 打印 simOut/日志结构
fprintf('\n--- simOut/日志结构 ---\n');
try
    fprintf('simOut 类型: %s\n', class(simOut));
    ds = [];
    % 1) 直接在顶层
    if isfield(simOut,'logsout')
        ds = simOut.logsout;
    end
    % 2) RL Toolbox 返回 experience 结构: SimulationInfo.SimulationOutput.logsout
    if isempty(ds) && isfield(simOut,'SimulationInfo')
        try
            ds = simOut.SimulationInfo.SimulationOutput.logsout;
        catch
        end
    end
    % 3) 其他包装形式
    if isempty(ds) && isfield(simOut,'simout')
        try
            ds = simOut.simout.logsout;
        catch
        end
    end

    if ~isempty(ds)
        fprintf('  logsout 类型: %s\n', class(ds));
        try
            nEl = ds.numElements;
            fprintf('  元素数: %d\n', nEl);
            for i = 1:nEl
                el = [];
                try
                    el = ds.getElement(i);
                catch
                    try
                        el = ds.get(i);
                    catch
                        el = [];
                    end
                end
                if ~isempty(el)
                    try


                        fprintf('    - %s\n', el.Name);
                    catch
                    end
                end
            end
        catch
        end
    else
        fprintf('  simOut 不含 logsout 字段\n');
    end
catch
end


% 若 logsout 为空，尝试从 SimulationOutput 中直接读取 To Workspace 变量
episodeData = struct();
try
    names = {'Battery_SOC','Battery_SOH','Battery_Power','P_batt','TotalCost'};
    for ii = 1:numel(names)
        nm = names{ii};
        if isfield(simOut, nm)
            val = simOut.(nm);
            if isnumeric(val)
                episodeData.(normalize_key(nm)) = to_timeseries_if_array(val, 3600);
            elseif isa(val,'timeseries')
                episodeData.(normalize_key(nm)) = val;
            end
            fprintf('  ✓ 从 SimulationOutput.%s 提取\n', nm);
        end
    end
catch
end

% 如已获得目标变量，直接保存并结束
if ~isempty(fieldnames(episodeData))
    try
        outDir = fullfile(repoRoot,'results','best_run');
        if ~exist(outDir,'dir'), mkdir(outDir); end
        save(fullfile(outDir,'best_episode_data.mat'), '-struct', 'episodeData', '-v7.3');
        fprintf('✓ 已保存提取数据到 %s（来自 SimulationOutput）\n', fullfile(outDir,'best_episode_data.mat'));
        fprintf('\n=== 诊断结束 ===\n');
        return;
    catch ME
        fprintf('⚠ 保存失败: %s\n', ME.message);
    end
end

% 尝试用 extract_best_episode 从 simOut 直接提取
try
    episodeData = extract_best_episode(simOut, [], [], maxSteps);

    fprintf('\n✓ 调用 extract_best_episode(simOut, ...) 完成\n');
catch ME
    fprintf('\n✗ extract_best_episode 失败: %s\n', ME.message);
    episodeData = struct();
end

% 保存提取结果（覆盖 best_run 下的数据）
try
    outDir = fullfile(repoRoot,'results','best_run');
    if ~exist(outDir,'dir'), mkdir(outDir); end
    save(fullfile(outDir,'best_episode_data.mat'), '-struct', 'episodeData', '-v7.3');
    fprintf('✓ 已保存提取数据到 %s\n', fullfile(outDir,'best_episode_data.mat'));
catch ME
    fprintf('⚠ 保存失败: %s\n', ME.message);
end

fprintf('\n=== 诊断结束 ===\n');
end



function project_root = find_project_root(start_dir)
%FIND_PROJECT_ROOT 自下而上查找同时包含 matlab/ 与 model/ 的目录
    project_root = start_dir;
    max_depth = 10;
    for i = 1:max_depth
        if exist(fullfile(project_root,'matlab'),'dir') && exist(fullfile(project_root,'model'),'dir')
            return;
        end
        parent_dir = fileparts(project_root);
        if isempty(parent_dir) || strcmp(parent_dir, project_root)
            break;
        end
        project_root = parent_dir;
    end
    error('RunManager:ProjectRootNotFound','无法从路径%s定位项目根目录', start_dir);
end



function print_struct_fields(s, name, depth)
%PRINT_STRUCT_FIELDS 递归打印结构体的字段与类型（受 depth 限制）
    if depth < 0, return; end
    try
        if isstruct(s)
            fns = fieldnames(s);
            fprintf('  [%s] 字段数: %d\n', name, numel(fns));

            for k = 1:numel(fns)
                fname = fns{k};
                try
                    val = s.(fname);
                    cls = class(val);
                catch
                    val = [];
                    cls = 'unknown';
                end
                fprintf('    - %s (%s)\n', fname, cls);
                % 递归一层
                if isstruct(val) && depth > 0
                    print_struct_fields(val, sprintf('%s.%s', name, fname), depth-1);
                end
            end
        else
            fprintf('  [%s] 类型: %s\n', name, class(s));
        end
    catch
    end
end



function mark_signals_for_logging(modelName, names)
%MARK_SIGNALS_FOR_LOGGING 批量按名称启用“Log signal data”
    try
        if ~bdIsLoaded(modelName), load_system(modelName); end
    catch
    end
    for i = 1:numel(names)
        nm = names{i};
        try
            hLines = find_system(modelName,'FindAll','on','Type','line','Name',nm);
            for k = 1:numel(hLines)
                h = hLines(k);
                try
                    Simulink.sdi.markSignalForStreaming(h,'on');
                catch
                    try, set_param(h,'DataLogging','on'); catch, end
                end
            end
        catch
        end
    end
end


function key = normalize_key(nm)
%NORMALIZE_KEY 
    s = lower(nm);
    if contains(s,'soc')
        key = 'Battery_SOC';
    elseif contains(s,'soh')
        key = 'Battery_SOH';
    elseif contains(s,'p_batt') || contains(s,'power') || contains(s,'batt')
        key = 'Battery_Power';
    elseif contains(s,'cost')
        key = 'TotalCost';
    else
        key = nm;
    end
end

function ts = to_timeseries_if_array(x, Ts)
%TO_TIMESERIES_IF_ARRAY 
    if isnumeric(x)
        x = x(:);
        t = (0:numel(x)-1)' .* Ts;
        try
            ts = timeseries(x, t);
        catch
            % Deep Learning Toolbox not required; ensure basic timeseries exists
            ts = timeseries(x, t);
        end
    else
        ts = x;
    end
end
