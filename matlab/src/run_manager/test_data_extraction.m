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
        try
            env = create_env();
            fprintf('✓ 使用 create_env() 构建 env\n');
        catch
        end
        if isempty(env)
            try
                env = build_env();
                fprintf('✓ 使用 build_env() 构建 env\n');
            catch
            end
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
        try
            modelName = env.mdl;
        catch
        end
    end
catch
end

if ~isempty(modelName)
    try
        if ~bdIsLoaded(modelName), load_system(modelName); end
    catch
    end
    try
        set_param(modelName,'FastRestart','off');
    catch
    end
    try
        set_param(modelName,'SignalLogging','on');
    catch
    end
    try
        set_param(modelName,'SignalLoggingName','logsout');
    catch
    end
    try
        set_param(modelName,'SignalLoggingSaveFormat','Dataset');
    catch
    end
    % 确保以“单一仿真输出”返回（将 logsout 等纳入 simOut）
    try
        set_param(modelName,'ReturnWorkspaceOutputs','on');
    catch
    end
    fprintf('DEBUG: 已关闭 FastRestart 并启用 SignalLogging + 单一仿真输出（模型：%s）\n', modelName);
else
    fprintf('DEBUG: 未能推断模型名，可能无法设置 FastRestart/SignalLogging\n');
end

% 打印模型 Signal Logging 配置参数
if ~isempty(modelName)
    try
        siglog = get_param(modelName, 'SignalLogging');
        siglogName = get_param(modelName, 'SignalLoggingName');
        siglogFmt  = get_param(modelName, 'SignalLoggingSaveFormat');
        singleOut  = get_param(modelName, 'ReturnWorkspaceOutputs');
        fprintf('Model SignalLogging: %s, Name: %s, Format: %s, SingleOut(ReturnWorkspaceOutputs): %s\n', siglog, siglogName, siglogFmt, singleOut);
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
            try
                nm = string(get_param(ln(k),'Name'));
            catch
            end
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


% 执行一次重放仿真（确保 RL Agent 变量名正确并可用）
try
    % 1) 定位 RL Agent 块与其参数“Agent”的变量名
    agentVarName = 'agent';
    agentBlock = '';
    try
        defaultBlk = [modelName '/RL Agent'];
        get_param(defaultBlk,'Handle'); % 若抛错则说明该路径不存在
        agentBlock = defaultBlk;
    catch
        try
            % 通过 MaskType 识别“Reinforcement Learning Agent”块
            blks = find_system(modelName, 'FollowLinks','on', 'LookUnderMasks','all', 'MaskType','Reinforcement Learning Agent');
            if ~isempty(blks)
                agentBlock = blks{1};
            end
        catch
        end
    end
    try
        if ~isempty(agentBlock)
            av = get_param(agentBlock, 'Agent');
            if isstring(av) || ischar(av)
                av = char(av);
                av = strtrim(av);
                if ~isempty(av)
                    agentVarName = av;
                end
            end
            fprintf('RL Agent 块: %s, Agent 变量名: %s\n', agentBlock, agentVarName);
        else
            fprintf('未定位 RL Agent 块，默认 Agent 变量名: %s\n', agentVarName);
        end
    catch ME1
        fprintf('读取 RL Agent 块参数失败: %s\n', ME1.message);
    end

    % 2) 将 agent 写入 base workspace 与 SimulationInput 变量表（双轨保障）
    try
        assignin('base', agentVarName, agent);
    catch ME2
        fprintf('assignin base %s 失败: %s\n', agentVarName, ME2.message);
    end
    if ~strcmp(agentVarName,'agent')
        try
            assignin('base','agent',agent); % 兼容部分模型默认名
        catch
        end
    end

    % 建议：批量标记目标信号为 Log（如用户已在模型中勾选，此步骤无副作用）
    try
        mark_signals_for_logging(modelName, {'Battery_SOC','Battery_SOH','P_batt','Battery_Power','TotalCost','SOH_Diff','SOC','SOH'});
    catch
    end

    % 3) 使用 SimulationInput 强制参数生效并返回 SimulationOutput
    try
        set_param(modelName,'FastRestart','off');
    catch
    end
    in = Simulink.SimulationInput(modelName);
    in = setModelParameter(in, 'FastRestart','off', 'ReturnWorkspaceOutputs','on', ...
        'SignalLogging','on', 'SignalLoggingName','logsout', 'SignalLoggingSaveFormat','Dataset');
    try
        in = setVariable(in, agentVarName, agent);
        if ~strcmp(agentVarName,'agent')
            in = setVariable(in, 'agent', agent); % 兼容别名
        end
    catch ME3
        fprintf('setVariable(%s) 失败: %s\n', agentVarName, ME3.message);
        fprintf('错误标识符: %s\n', ME3.identifier);
        fprintf('错误堆栈:\n');
        for ii = 1:length(ME3.stack)
            fprintf('  文件: %s, 函数: %s, 行: %d\n', ME3.stack(ii).file, ME3.stack(ii).name, ME3.stack(ii).line);
        end
    end

    % 4) 诊断：确认 base workspace 是否已有对应变量
    try
        hasVar = evalin('base', sprintf('exist(''%s'',''var'')', agentVarName));
        fprintf('Base workspace 存在变量 %s? %d\n', agentVarName, hasVar);
    catch
    end

    % 5) 运行仿真前打印 SimulationInput.Variables
    try
        vars = in.Variables;
        nVars = numel(vars);
        fprintf('SimulationInput.Variables 数量: %d\n', nVars);
        if nVars > 0
            names = string({vars.Name});
            fprintf('SimulationInput.Variables 名称: %s\n', strjoin(cellstr(names), ', '));
            hasAgentVar = any(strcmp(cellstr(names), agentVarName));
            fprintf('包含 %s? %d\n', agentVarName, hasAgentVar);
        end
    catch MEv
        fprintf('读取 SimulationInput.Variables 失败: %s\n', MEv.message);
    end

    % 5a) 数据源检查与注入（From Workspace/Constant 依赖）
    try
        dataFile = fullfile('matlab','src','microgrid_simulation_data.mat');
        % 3 个 From Workspace 块及其 VariableName
        fwBlocks = {
            'Microgrid/load_power_profile', ...
            'Microgrid/price_profile', ...
            'Microgrid/pv_power_profile' ...
        };
        for bb = 1:numel(fwBlocks)
            blk = fwBlocks{bb};
            try
                vName = '';
                try
                    vName = char(get_param(blk,'VariableName'));
                catch
                end
                vName = strtrim(vName);
                if ~isempty(vName)
                    fprintf('  依赖块: %s, VariableName: %s\n', blk, vName);
                    % 若 base 不存在，则尝试从数据文件加载并注入
                    hasBase = 0;
                    try
                        hasBase = evalin('base', sprintf('exist(''%s'',''var'')', vName));
                    catch
                    end
                    if ~hasBase
                        if exist(dataFile,'file') == 2
                            try
                                S = load(dataFile, vName);
                                if isfield(S, vName)
                                    assignin('base', vName, S.(vName));
                                    try
                                        in = setVariable(in, vName, S.(vName));
                                    catch
                                    end
                                    fprintf('    -> 已从数据文件注入 %s (类型: %s)。\n', vName, class(S.(vName)));
                                else
                                    fprintf('    -> 数据文件中未包含变量 %s。\n', vName);
                                end
                            catch MEld
                                fprintf('    -> 加载数据文件变量 %s 失败: %s\n', vName, MEld.message);
                            end
                        else
                            fprintf('    -> 未找到数据文件: %s\n', dataFile);
                        end
                    end
                else
                    fprintf('  依赖块: %s, 未配置 VariableName。\n', blk);
                end
            catch ME_fw
                fprintf('  读取 From Workspace 配置失败(%s): %s\n', blk, ME_fw.message);
            end
        end
        % Constant 块（可能引用变量）
        constBlk = 'Microgrid/Hierarchical Reward System/Constant';
        try
            valStr = char(get_param(constBlk,'Value'));
            valStr = strtrim(valStr);
            fprintf('  Constant 块: %s, Value: %s\n', constBlk, valStr);
            % 如果 Value 是变量名且 base 不存在，尝试从数据文件注入
            isNum = ~isnan(str2double(valStr));
            if ~isNum
                looksVar = isvarname(valStr);
                if looksVar
                    hasBase = 0;
                    try
                        hasBase = evalin('base', sprintf('exist(''%s'',''var'')', valStr));
                    catch
                    end
                    if ~hasBase && exist(dataFile,'file') == 2
                        try
                            S2 = load(dataFile, valStr);
                            if isfield(S2, valStr)
                                assignin('base', valStr, S2.(valStr));
                                try
                                    in = setVariable(in, valStr, S2.(valStr));
                                catch
                                end
                                fprintf('    -> 已从数据文件注入 %s (类型: %s)。\n', valStr, class(S2.(valStr)));
                            else
                                fprintf('    -> 数据文件中未包含变量 %s。\n', valStr);
                                if strcmp(valStr,'g_episode_num')
                                    try
                                        assignin('base','g_episode_num',1);
                                        in = setVariable(in,'g_episode_num',1);
                                        fprintf('    -> 变量 g_episode_num 不存在，已注入默认值 1（用于单次仿真测试）。\n');
                                    catch
                                    end
                                end
                            end
                        catch MEld2
                            fprintf('    -> 加载数据文件变量 %s 失败: %s\n', valStr, MEld2.message);
                        end
                    end
                        % 兜底：若依旧缺失且为 g_episode_num，则注入默认值 1
                        try
                            hasBase2 = 0;
                            try
                                hasBase2 = evalin('base', sprintf('exist(''%s'',''var'')', valStr));
                            catch
                            end
                            if ~hasBase2 && strcmp(valStr,'g_episode_num')
                                try
                                    assignin('base','g_episode_num',1);
                                    in = setVariable(in,'g_episode_num',1);
                                    fprintf('    -> 变量 g_episode_num 仍缺失，已兜底注入默认值 1（用于单次仿真测试）。\n');
                                catch
                                end
                            end
                        catch
                        end

                else
                    % 可能是表达式，尝试一次解析仅用于提示
                    try
                        evalin('base', valStr);
                    catch
                        fprintf('    -> Constant.Value 看似表达式且无法在 base 解析，请检查。\n');
                    end
                end
                    % 统一兜底：确保 g_episode_num 在 base 与 SimulationInput 中都存在
                    try
                        hasGEpi = evalin('base','exist(''g_episode_num'',''var'')');
                    catch
                        hasGEpi = 0;
                    end
                    if ~hasGEpi
                        % base 缺失：注入默认值 1 并同步到 SimulationInput
                        try
                            assignin('base','g_episode_num',1);
                            in = setVariable(in,'g_episode_num',1);
                            fprintf('    -> 兜底：已注入默认 g_episode_num=1（用于单次仿真测试）。\n');
                        catch
                        end
                    else
                        % base 已存在：确保 SimulationInput 也包含该变量，并打印其值
                        try
                            geVal = evalin('base','g_episode_num');
                        catch
                            geVal = [];
                        end
                        try
                            vars = in.Variables; names = string({vars.Name});
                        catch
                            names = string.empty;
                        end
                        if isempty(names) || ~any(strcmp(names,'g_episode_num'))
                            try
                                in = setVariable(in,'g_episode_num',geVal);
                                fprintf('    -> 诊断：base 已存在 g_episode_num，已同步到 SimulationInput。\n');
                            catch
                            end
                        else
                            fprintf('    -> 诊断：g_episode_num 已存在于 base 与 SimulationInput。\n');
                        end
                        % 尝试打印其数值（若为标量数值）
                        try
                            if isnumeric(geVal) && isscalar(geVal)
                                fprintf('       g_episode_num 值 = %g\n', geVal);
                            else
                                fprintf('       g_episode_num 类型 = %s\n', class(geVal));
                            end
                        catch
                        end
                    end

            end
        catch MEc
            fprintf('  读取 Constant 配置失败: %s\n', MEc.message);
        end
        % 刷新一次 Variables 列表（可见刚注入的变量）
        try
            vars = in.Variables;
            nVars = numel(vars);
            fprintf('  注入后 Variables 数量: %d\n', nVars);
            if nVars > 0
                names = string({vars.Name});
                fprintf('  注入后 Variables 名称: %s\n', strjoin(cellstr(names), ', '));
            end
        catch
        end
    catch MEdat
        fprintf('数据依赖注入阶段出现异常: %s\n', MEdat.message);
    end

    % 5) 运行仿真
    simOut = sim(in);
catch ME
    fprintf('\n✗ 直接 sim(modelName) 失败: %s\n', ME.message);
    fprintf('错误标识符: %s\n', ME.identifier);
    fprintf('错误堆栈:\n');
    for ii = 1:length(ME.stack)
        fprintf('  文件: %s, 函数: %s, 行: %d\n', ME.stack(ii).file, ME.stack(ii).name, ME.stack(ii).line);
    end
    % 展开 MultipleErrors 异常
    if strcmp(ME.identifier, 'MATLAB:MException:MultipleErrors')
        fprintf('\n=== 展开 MultipleErrors（共 %d 个子错误）===\n', length(ME.cause));
        for jj = 1:length(ME.cause)
            subME = ME.cause{jj};
            fprintf('\n--- 子错误 %d/%d ---\n', jj, length(ME.cause));
            fprintf('  标识符: %s\n', subME.identifier);
            fprintf('  消息: %s\n', subME.message);
            fprintf('  堆栈:\n');
            for kk = 1:length(subME.stack)
                fprintf('    文件: %s, 函数: %s, 行: %d\n', subME.stack(kk).file, subME.stack(kk).name, subME.stack(kk).line);
            end
        end
    end
    try
        if exist('agentVarName','var') && ~isempty(agentVarName)
            hasVar = evalin('base', sprintf('exist(''%s'',''var'')', agentVarName));
            fprintf('  诊断: Base 是否存在 %s? %d\n', agentVarName, hasVar);
        end
    catch
    end
    return;
end


% 打印 simOut/日志结构
fprintf('\n--- simOut/日志结构 ---\n');
try
    fprintf('simOut 类型: %s\n', class(simOut));
    ds = [];

    % 优先使用 SimulationOutput 的 get 接口（适配对象动态属性）
    try
        ds = simOut.get('logsout');
    catch
    end
    % 直接点取（若 get 不支持/未创建则可能抛异常）
    if isempty(ds)
        try
            ds = simOut.logsout;
        catch
        end
    end
    % 兼容 RL 经验返回格式
    if isempty(ds)
        try
            ds = simOut.SimulationInfo.SimulationOutput.logsout;
        catch
        end
    end
    % 其他包装形式
    if isempty(ds)
        try
            ds = simOut.simout.logsout;
        catch
        end
    end
    % 最后尝试 get(simOut) 返回的结构体中探测
    if isempty(ds)
        try
            sAll = get(simOut);
            if isstruct(sAll) && isfield(sAll,'logsout')
                ds = sAll.logsout;
            end
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
        % 仅保存白名单字段，避免将整个工作区或无关字段写入 MAT 文件
        allowed = {'Battery_SOC','Battery_SOH','Battery_Power','P_batt','TotalCost','SOH_Diff','SOH_diff','episode_reward'};
        dataToSave = struct();
        for kk = 1:numel(allowed)
            fld = allowed{kk};
            if isfield(episodeData, fld)
                dataToSave.(fld) = episodeData.(fld);
            end
        end
        dataPath = fullfile(outDir,'best_episode_data.mat');
        tempPath = [dataPath, '.tmp'];
        if exist(tempPath,'file'), delete(tempPath); end
        save(tempPath, '-struct', 'dataToSave', '-v7.3');
        try
            movefile(tempPath, dataPath, 'f');
        catch
            movefile(tempPath, dataPath);
        end
        fprintf('✓ 已保存提取数据到 %s（来自 SimulationOutput，仅白名单字段）\n', dataPath);
        % 立即验证文件内容
        try
            vars = whos('-file', dataPath);
            names = string({vars.name});
            fprintf('  文件变量：%s\n', strjoin(cellstr(names), ', '));
        catch
        end
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
    % 仅保存白名单字段，避免将整个工作区或无关字段写入 MAT 文件
    allowed = {'Battery_SOC','Battery_SOH','Battery_Power','P_batt','TotalCost','SOH_Diff','SOH_diff','episode_reward'};
    dataToSave = struct();
    for kk = 1:numel(allowed)
        fld = allowed{kk};
        if isfield(episodeData, fld)
            dataToSave.(fld) = episodeData.(fld);
        end
    end
    dataPath = fullfile(outDir,'best_episode_data.mat');
    tempPath = [dataPath, '.tmp'];
    if exist(tempPath,'file'), delete(tempPath); end
    save(tempPath, '-struct', 'dataToSave', '-v7.3');
    try
        movefile(tempPath, dataPath, 'f');
    catch
        movefile(tempPath, dataPath);
    end
    fprintf('✓ 已保存提取数据到 %s\n', dataPath);
    % 立即验证文件内容
    try
        vars = whos('-file', dataPath);
        names = string({vars.name});
        fprintf('  文件变量：%s\n', strjoin(cellstr(names), ', '));
    catch
    end
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
                    try
                        set_param(h,'DataLogging','on');
                    catch
                    end
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
