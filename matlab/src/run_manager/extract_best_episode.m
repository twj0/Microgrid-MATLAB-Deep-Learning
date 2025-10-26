function episodeData = extract_best_episode(simOut, agent, env, maxSteps)
%EXTRACT_BEST_EPISODE 从仿真结果或重新运行提取最优episode数据
%   只保存必要的时序数据(SOC/SOH/Power)，减小文件体积
%
% 输入:
%   simOut - Simulink仿真输出(可选,如果为空则重新运行)
%   agent - 训练好的智能体(重新运行时需要)
%   env - 仿真环境(重新运行时需要)
%   maxSteps - 最大步数(默认720)
%
% 输出:
%   episodeData - 结构体,包含Battery_SOC, Battery_SOH, Battery_Power等

    if nargin < 4
        maxSteps = 720;
    end

    episodeData = struct(); %#ok<NASGU> % 初始化默认输出，兼容无输入时的回退路径

    % 情况1: 提供了simOut,直接提取数据
    if nargin >= 1 && ~isempty(simOut)
        fprintf('\n提取episode数据...\n');
        episodeData = extract_from_simout_clean_v2(simOut);

    % 情况2: 没有simOut,需要重新运行仿真
    elseif nargin >= 3 && ~isempty(agent) && ~isempty(env)
        fprintf('\n重新运行仿真以提取数据...\n');
        % —— 诊断与配置：尝试关闭 Fast Restart，开启 Signal Logging ——
        try
            modelName = '';
            % 推断 Simulink 模型名（兼容不同 env 实现）
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
                    % 忽略
                end
            end
            if ~isempty(modelName)
                try
                    if ~bdIsLoaded(modelName)
                        load_system(modelName);
                    end
                catch
                end
                % 关闭 Fast Restart，开启 Signal Logging 到 logsout（Dataset）
                try
                    set_param(modelName, 'FastRestart', 'off');
                catch
                end
                try
                    set_param(modelName, 'SignalLogging', 'on');
                catch
                end
                try
                    set_param(modelName, 'SignalLoggingName', 'logsout');
                catch
                end
                try
                    set_param(modelName, 'SignalLoggingSaveFormat', 'Dataset');
                catch
                end
                fprintf('  DEBUG: 已配置模型 %s 关闭 FastRestart 并启用 SignalLogging\n', modelName);
            else
                fprintf('  DEBUG: 未能从 env 推断模型名，跳过 FastRestart/SignalLogging 配置\n');
            end
        catch ME
            fprintf('  DEBUG: 设置 FastRestart/SignalLogging 失败: %s\n', ME.message);
        end


        fprintf('  ⏱ 这可能需要几分钟时间...\n');

        % 方案A（主路径）：两输出 sim(env,agent) 获取 simInfo.SimulationOutput
        try
            simOpts = rlSimulationOptions('MaxSteps', maxSteps, 'UseParallel', false);
            fprintf('  路径A: sim(env,agent) 双输出以获取 SimulationOutput...\n');
            [~, simInfo] = sim(env, agent, simOpts);
            try
                simOut = simInfo.SimulationOutput;
            catch
                error('simInfo 不含 SimulationOutput 字段');
            end
            episodeData = extract_from_simout_clean_v2(simOut);
        catch ME_A
            % 方案B（兜底）：使用 SimulationInput 进行 model 级仿真，并确保 RL Agent 变量正确注入
            fprintf('  路径A失败，切换到路径B（model 级 sim）：%s\n', ME_A.message);
            if isempty(modelName)
                error('无法推断模型名，且路径A失败，终止提取。');
            end

            % 定位 RL Agent 块与其参数“Agent”的变量名
            agentVarName = 'agent';
            agentBlock = '';
            try
                defaultBlk = [modelName '/RL Agent'];
                get_param(defaultBlk,'Handle');
                agentBlock = defaultBlk;
            catch
                try
                    blks = find_system(modelName,'FollowLinks','on','LookUnderMasks','all','MaskType','Reinforcement Learning Agent');
                    if ~isempty(blks)
                        agentBlock = blks{1};
                    end
                catch
                end
            end
            try
                if ~isempty(agentBlock)
                    av = get_param(agentBlock,'Agent');
                    if isstring(av) || ischar(av)
                        av = strtrim(char(av));
                        if ~isempty(av)
                            agentVarName = av;
                        end
                    end
                    fprintf('  路径B: RL Agent 块: %s, Agent 变量名: %s\n', agentBlock, agentVarName);
                else
                    fprintf('  路径B: 未定位 RL Agent 块，默认 Agent 变量名: %s\n', agentVarName);
                end
            catch MEB1
                fprintf('  路径B: 读取 RL Agent 参数失败: %s\n', MEB1.message);
            end

            % 使用 SimulationInput 配置模型参数与变量
            try
                if ~bdIsLoaded(modelName), load_system(modelName); end
            catch
            end
            in = Simulink.SimulationInput(modelName);
            in = setModelParameter(in, 'FastRestart','off', 'ReturnWorkspaceOutputs','on', ...
                'SignalLogging','on', 'SignalLoggingName','logsout', 'SignalLoggingSaveFormat','Dataset');
            try
                in = setVariable(in, agentVarName, agent);
                if ~strcmp(agentVarName,'agent')
                    in = setVariable(in, 'agent', agent); % 兼容常见别名
                end
            catch MEB2
                fprintf('  路径B: setVariable(%s) 失败: %s\n', agentVarName, MEB2.message);
                fprintf('  标识符: %s\n', MEB2.identifier);
                fprintf('  堆栈:\n');
                for ii = 1:length(MEB2.stack)
                    fprintf('    文件: %s, 函数: %s, 行: %d\n', MEB2.stack(ii).file, MEB2.stack(ii).name, MEB2.stack(ii).line);
                end
            end

            % 同步写入 base workspace（双重保障）
            try
                assignin('base', agentVarName, agent);
            catch
                % 忽略写入 base workspace 失败
            end
            if ~strcmp(agentVarName,'agent')
                try
                    assignin('base','agent',agent);
                catch
                    % 忽略写入 base workspace 失败（别名）
                end
            end

            fprintf('  路径B: 使用 SimulationInput 运行仿真...\n');
            % 诊断：打印 SimulationInput.Variables
            try
                vars = in.Variables;
                nVars = numel(vars);
                fprintf('  路径B: SimulationInput.Variables 数量: %d\n', nVars);
                if nVars > 0
                    names = string({vars.Name});
                    fprintf('  路径B: Variables 名称: %s\n', strjoin(cellstr(names), ', '));
                    hasAgentVar = any(strcmp(cellstr(names), agentVarName));
                    fprintf('  路径B: 是否包含 %s? %d\n', agentVarName, hasAgentVar);
                end
            catch MEv
                fprintf('  路径B: 读取 SimulationInput.Variables 失败: %s\n', MEv.message);
            end
            % 路径B: 数据源检查与注入（From Workspace/Constant 依赖）
            try
                dataFile = fullfile('matlab','src','microgrid_simulation_data.mat');
                fwBlocks = {
                    [modelName '/load_power_profile'], ...
                    [modelName '/price_profile'], ...
                    [modelName '/pv_power_profile'] ...
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
                            fprintf('  路径B: 依赖块: %s, VariableName: %s\n', blk, vName);
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
                            fprintf('  路径B: 依赖块: %s, 未配置 VariableName。\n', blk);
                        end
                    catch ME_fw
                        fprintf('  路径B: 读取 From Workspace 配置失败(%s): %s\n', blk, ME_fw.message);
                    end
                end
                % Constant 块（可能引用变量）
                constBlk = [modelName '/Hierarchical Reward System/Constant'];
                try
                    valStr = char(get_param(constBlk,'Value'));
                    valStr = strtrim(valStr);
                    fprintf('  路径B: Constant 块: %s, Value: %s\n', constBlk, valStr);
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
                            %

                            %

% 
                            %
% 
                            %
%
%
%
%
%
%
%
%
%
%
%
%
%
%

                            %
%
% Fallback: ensure g_episode_num default injection if still missing
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
                                            catch
                                            end
                                        end
                                    end
                                catch MEld2
                                    fprintf('    -> 加载数据文件变量 %s 失败: %s\n', valStr, MEld2.message);
                                end
                            end
                        else
                            try
                                evalin('base', valStr);
                            catch
                                fprintf('    -> Constant.Value 看似表达式且无法在 base 解析，请检查。\n');
                            end
                        end
                    end
                catch MEc
                    fprintf('  路径B: 读取 Constant 配置失败: %s\n', MEc.message);
                end
                % 刷新一次 Variables 列表
                try
                    vars = in.Variables;
                    nVars = numel(vars);
                    fprintf('  路径B: 注入后 Variables 数量: %d\n', nVars);
                    if nVars > 0
                        names = string({vars.Name});
                        fprintf('  路径B: 注入后 Variables 名称: %s\n', strjoin(cellstr(names), ', '));
                    end
                catch
                end
            catch MEdat
                fprintf('  路径B: 数据依赖注入阶段出现异常: %s\n', MEdat.message);
            end

            try
                simOut = sim(in);
            catch ME_B
                fprintf('  路径B: sim(in) 失败: %s\n', ME_B.message);
                fprintf('  标识符: %s\n', ME_B.identifier);
                fprintf('  堆栈:\n');
                for ii = 1:length(ME_B.stack)
                    fprintf('    文件: %s, 函数: %s, 行: %d\n', ME_B.stack(ii).file, ME_B.stack(ii).name, ME_B.stack(ii).line);
                end
                % 展开 MultipleErrors 异常
                if strcmp(ME_B.identifier, 'MATLAB:MException:MultipleErrors')
                    fprintf('\n=== 展开 MultipleErrors（共 %d 个子错误）===\n', length(ME_B.cause));
                    for jj = 1:length(ME_B.cause)
                        subME = ME_B.cause{jj};
                        fprintf('\n--- 子错误 %d/%d ---\n', jj, length(ME_B.cause));
                        fprintf('  标识符: %s\n', subME.identifier);
                        fprintf('  消息: %s\n', subME.message);
                        fprintf('  堆栈:\n');
                        for kk = 1:length(subME.stack)
                            fprintf('    文件: %s, 函数: %s, 行: %d\n', subME.stack(kk).file, subME.stack(kk).name, subME.stack(kk).line);
                        end
                    end
                end
                rethrow(ME_B);
            end
            episodeData = extract_from_simout_clean_v2(simOut);
        end

    else
        error('extract_best_episode需要提供simOut或(agent + env)');
    end

    % 添加元信息
    episodeData.timestamp = datetime('now');
    episodeData.description = '最优episode仿真数据(已降采样至每小时)';
    episodeData.extraction_time = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));

    fprintf('  ✓ 数据提取完成\n');

    % 显示数据统计
    display_data_summary(episodeData);
end

function data = extract_from_simout(simOut)
    %EXTRACT_FROM_SIMOUT 从simOut结构体提取关键信号

    data = struct();

    % 尝试从logsout提取
    if isfield(simOut, 'logsout')
        data = extract_from_logsout(simOut.logsout, data);
    end

    % 尝试从yout提取(Simulink R2018b及更早版本)
    if isfield(simOut, 'yout') && isempty(fieldnames(data))
        fprintf('  ⚠ logsout为空,尝试从yout提取...\n');
        data = extract_from_yout(simOut.yout, data);
    end

    % 尝试从workspace变量提取(最后手段)
    if isempty(fieldnames(data))
        fprintf('  ⚠ simOut未包含数据,尝试从workspace提取...\n');
        data = extract_from_workspace(data);
    end

    % 屏蔽受损的 DEBUG 代码块（无影响提取逻辑）
    if local_debug_enabled()

    % DEBUG: 
    try
        if isa(logsout,'Simulink.SimulationData.Dataset')
            fprintf('  DEBUG: logsout 元素: %d\n', logsout.numElements);
            for jj = 1:logsout.numElements
                try
                    e = logsout.getElement(jj);
                catch
                    e = logsout.get(jj);
                end
                try
                    fprintf('    - %s\n', e.Name);
                catch
                    % ignore
                end
            end
        end

    catch
    end


    end

    % 降采样以减小文件大小
    if ~isempty(fieldnames(data))
        data = downsample_data(data);
    end
end

function data = extract_from_logsout(logsout, data)
    %EXTRACT_FROM_LOGSOUT 从logsout提取信号

    % 暂时关闭 getElement 未命名命中时的啰嗦警告，函数退出自动恢复
    try
        warnId = 'Simulink:SimulationData:DatasetElementNotFound';
        stWarn = warning('query', warnId);
        prevState = stWarn.state;
        warning('off', warnId);
        cleanupWarn = onCleanup(@() warning(prevState, warnId));
    catch
    end

    % 必须信号: Battery_SOC
    % DEBUG: 列出 logsout 中可用信号名称，便于诊断命名不一致
    try
        if isa(logsout,'Simulink.SimulationData.Dataset')
            nEl = logsout.numElements;
            fprintf('  DEBUG: logsout 元素列表（共 %d）:\n', nEl);
            for ii = 1:nEl
                el = [];
                try
                    el = logsout.getElement(ii);
                catch
                    try
                        el = logsout.get(ii);
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
        end
    catch
    end

    % Battery_SOC with alternate names and fuzzy match
    try
        socNames = {'Battery_SOC','battery_soc','BatterySOC','SOC','soc'};
        soc = [];
        socTS = [];
        for iN = 1:numel(socNames)
            try
                nameKey = socNames{iN};
                if isa(logsout,'Simulink.SimulationData.Dataset')
                    for ii = 1:logsout.numElements
                        el = [];
                        try
    el = logsout.getElement(ii);
catch
    el = [];
end
                        if isempty(el), continue; end
                        nm1 = '';
                        try
    nm1 = string(el.Name);
catch
    nm1 = '';
end
                        if strcmpi(nm1, nameKey)
                            if isprop(el,'Values') && isa(el.Values,'timeseries')
                                soc = el; socTS = el.Values; break;
                            elseif isprop(el,'Values') && isa(el.Values,'Simulink.SimulationData.Dataset')
                                val = el.Values;
                                for jj = 1:val.numElements
                                    sub = [];
                                    try
    sub = val.getElement(jj);
catch
    sub = [];
end
                                    if ~isempty(sub) && isprop(sub,'Values') && isa(sub.Values,'timeseries')
                                        soc = sub; socTS = sub.Values; break;
                                    end
                                end
                                if ~isempty(socTS), break; end
                            end
                        end
                    end
                    if ~isempty(socTS), break; end
                end
            catch
            end
        end
        if isempty(socTS)
            % fallback: scan elements for name containing 'soc' (Name or BlockPath)
            if isa(logsout,'Simulink.SimulationData.Dataset')
                for ii = 1:logsout.numElements
                    el = [];
                    try
                        el = logsout.getElement(ii);
                    catch
                        el = [];
                    end
                    if isempty(el), continue; end
                    try
                        nm1 = '';
                        try
                            nm1 = lower(string(el.Name));
                        catch
                        end
                        nm2 = '';
                        try
                            bp = el.BlockPath;
                            try
                                nm2 = lower(char(bp));
                            catch
                                nm2 = '';
                            end
                        catch
                            nm2 = '';
                        end
                        if contains(nm1,'soc') || contains(nm2,'soc')
                            % Try to extract timeseries directly or from nested Dataset
                            if isprop(el,'Values') && isa(el.Values,'timeseries')
                                soc = el; socTS = el.Values; break;
                            elseif isprop(el,'Values') && isa(el.Values,'Simulink.SimulationData.Dataset')
                                val = el.Values;
                                for jj = 1:val.numElements
                                    sub = [];
                                    try
    sub = val.getElement(jj);
catch
    sub = [];
end
                                    if ~isempty(sub) && isprop(sub,'Values') && isa(sub.Values,'timeseries')
                                        soc = sub; socTS = sub.Values; break;
                                    end
                                end
                                if ~isempty(socTS), break; end
                            end
                        end
                    catch
                    end
                end
            end
        end
        if ~isempty(socTS)
            data.Battery_SOC = timeseries(socTS.Data, socTS.Time);
            fprintf('  ✓ 提取Battery_SOC: %d数据点\n', length(socTS.Data));
        else
            error('Battery_SOC not found in logsout');
        end
    catch ME
        fprintf('  ⚠ 未找到Battery_SOC: %s\n', ME.message);
    end

    % 必须信号: Battery_SOH
    % Battery_SOH with alternate names and fuzzy match
    try
        sohNames = {'Battery_SOH','battery_soh','BatterySOH','SOH','soh'};
        soh = [];
        sohTS = [];
        for iN = 1:numel(sohNames)
            try
                nameKey = sohNames{iN};
                if isa(logsout,'Simulink.SimulationData.Dataset')
                    for ii = 1:logsout.numElements
                        el = [];
                        try
    el = logsout.getElement(ii);
catch
    el = [];
end
                        if isempty(el), continue; end
                        nm1 = '';
                        try
    nm1 = string(el.Name);
catch
    nm1 = '';
end
                        if strcmpi(nm1, nameKey)
                            if isprop(el,'Values') && isa(el.Values,'timeseries')
                                soh = el; sohTS = el.Values; break;
                            elseif isprop(el,'Values') && isa(el.Values,'Simulink.SimulationData.Dataset')
                                val = el.Values;
                                for jj = 1:val.numElements
                                    sub = [];
                                    try
    sub = val.getElement(jj);
catch
    sub = [];
end
                                    if ~isempty(sub) && isprop(sub,'Values') && isa(sub.Values,'timeseries')
                                        soh = sub; sohTS = sub.Values; break;
                                    end
                                end
                                if ~isempty(sohTS), break; end
                            end
                        end
                    end
                    if ~isempty(sohTS), break; end
                end
            catch
            end
        end
        if isempty(sohTS)
            if isa(logsout,'Simulink.SimulationData.Dataset')
                for ii = 1:logsout.numElements
                    el = [];
                    try
                        el = logsout.getElement(ii);
                    catch
                        el = [];
                    end
                    if isempty(el), continue; end
                    try
                        nm1 = '';
                        try
                            nm1 = lower(string(el.Name));
                        catch
                        end
                        nm2 = '';
                        try
                            bp = el.BlockPath;
                            try
                                nm2 = lower(char(bp));
                            catch
                                nm2 = '';
                            end
                        catch
                            nm2 = '';
                        end
                        if contains(nm1,'soh') || contains(nm2,'soh')
                            if isprop(el,'Values') && isa(el.Values,'timeseries')
                                soh = el; sohTS = el.Values; break;
                            elseif isprop(el,'Values') && isa(el.Values,'Simulink.SimulationData.Dataset')
                                val = el.Values;
                                for jj = 1:val.numElements
                                    sub = [];
                                    try
    sub = val.getElement(jj);
catch
    sub = [];
end
                                    if ~isempty(sub) && isprop(sub,'Values') && isa(sub.Values,'timeseries')
                                        soh = sub; sohTS = sub.Values; break;
                                    end
                                end
                                if ~isempty(sohTS), break; end
                            end
                        end
                    catch
                    end
                end
            end
        end
        if ~isempty(sohTS)
            data.Battery_SOH = timeseries(sohTS.Data, sohTS.Time);
            fprintf('  ✓ 提取Battery_SOH: %d数据点\n', length(sohTS.Data));
        else
            error('Battery_SOH not found in logsout');
        end
    catch ME
        fprintf('  ⚠ 未找到Battery_SOH: %s\n', ME.message);
    end

    % 可选信号: Battery_Power（含别名 P_batt 等）
    try
        pwrNames = {'Battery_Power','battery_power','Batt_Power','BatteryPower','Power_Battery', ...
                    'P_batt','P_Batt','P_battery','P_batt_W'};
        power = [];
        for iN = 1:numel(pwrNames)
            try
                nameKey = pwrNames{iN};
                if isa(logsout,'Simulink.SimulationData.Dataset')
                    for ii = 1:logsout.numElements
                        el = [];
                        try
    el = logsout.getElement(ii);
catch
    el = [];
end
                        if isempty(el), continue; end
                        nm1 = '';
                        try
    nm1 = string(el.Name);
catch
    nm1 = '';
end
                        if strcmpi(nm1, nameKey)
                            if isprop(el,'Values') && isa(el.Values,'timeseries')
                                power = el; break;
                            elseif isprop(el,'Values') && isa(el.Values,'Simulink.SimulationData.Dataset')
                                val = el.Values;
                                for jj = 1:val.numElements
                                    sub = [];
                                    try
    sub = val.getElement(jj);
catch
    sub = [];
end
                                    if ~isempty(sub) && isprop(sub,'Values') && isa(sub.Values,'timeseries')
                                        power = sub; break;
                                    end
                                end
                                if ~isempty(power) && isprop(power,'Values') && isa(power.Values,'timeseries')
                                    break;
                                end
                            end
                        end
                    end
                    if ~isempty(power) && isprop(power,'Values') && isa(power.Values,'timeseries')
                        break;
                    end
                end
            catch
            end
        end
        if isempty(power) && isa(logsout,'Simulink.SimulationData.Dataset')
            for ii = 1:logsout.numElements
                el = [];
                try
                    el = logsout.getElement(ii);
                catch
                    el = [];
                end
                if isempty(el), continue; end
                try
                    nm1 = '';
                    try
                        nm1 = lower(string(el.Name));
                    catch
                    end
                    nm2 = '';
                    try
                        bp = el.BlockPath;
                        try
                            nm2 = lower(char(bp));
                        catch
                            nm2 = '';
                        end
                    catch
                        nm2 = '';
                    end
                    % 匹配电池功率：兼容 'power'/'p_batt'/'p...' + 'batt' 组合，优先 timeseries，Dataset 则向下展开
                    isBatt1 = contains(nm1,'batt') || contains(nm1,'battery');
                    isBatt2 = contains(nm2,'batt') || contains(nm2,'battery');
                    hasPow1  = contains(nm1,'power') || contains(nm1,'p_batt') || startsWith(strtrim(nm1),'p');
                    hasPow2  = contains(nm2,'power') || contains(nm2,'p_batt') || startsWith(strtrim(nm2),'p');
                    if (isBatt1 || isBatt2) && (hasPow1 || hasPow2)
                        if isprop(el,'Values') && isa(el.Values,'timeseries')
                            power = el; break;
                        elseif isprop(el,'Values') && isa(el.Values,'Simulink.SimulationData.Dataset')
                            val = el.Values;
                            for jj = 1:val.numElements
                                sub = [];
                                try
    sub = val.getElement(jj);
catch
    sub = [];
end
                                if ~isempty(sub) && isprop(sub,'Values') && isa(sub.Values,'timeseries')
                                    power = sub; break;
                                end
                            end
                            if ~isempty(power) && isprop(power,'Values') && isa(power.Values,'timeseries')
                                break;
                            end
                        end
                    end
                catch
                end
            end
        end
        if ~isempty(power) && isprop(power,'Values') && isa(power.Values,'timeseries')
            data.Battery_Power = timeseries(power.Values.Data, power.Values.Time);
            fprintf('  ✓ 提取Battery_Power: %d数据点\n', length(power.Values.Data));
        end
    catch
        % Battery_Power可选,不报错
    end

    % 可选信号: Grid_Cost (用于经济分析) - 安全扫描，无警告
    try
        costTS = [];
        if isa(logsout,'Simulink.SimulationData.Dataset')
            targetNames = {'grid_cost','gridcost','cost_grid'};
            for ii = 1:logsout.numElements
                el = [];
                try
    el = logsout.getElement(ii);
catch
    el = [];
end
                if isempty(el), continue; end
                nm1 = '';
                try
    nm1 = string(el.Name);
catch
    nm1 = '';
end
                if any(strcmpi(nm1, targetNames))
                    if isprop(el,'Values') && isa(el.Values,'timeseries')
                        costTS = el.Values; break;
                    elseif isprop(el,'Values') && isa(el.Values,'Simulink.SimulationData.Dataset')
                        val = el.Values;
                        for jj = 1:val.numElements
                            sub = [];
try
    sub = val.getElement(jj);
catch
    sub = [];
end
                            if ~isempty(sub) && isprop(sub,'Values') && isa(sub.Values,'timeseries')
                                costTS = sub.Values; break;
                            end
                        end
                        if ~isempty(costTS), break; end
                    end
                end
            end
        end
        if ~isempty(costTS)
            data.cumulative_cost = sum(costTS.Data, 'omitnan');
            data.average_hourly_cost = mean(costTS.Data, 'omitnan');
            fprintf('  ✓ 提取Grid_Cost: 累计成本 = %.2f\n', data.cumulative_cost);
        end
    catch
        % Grid_Cost可选
    end
    % 可选信号: TotalCost (总成本)
    try
        tc = logsout.getElement('TotalCost');
        if isprop(tc,'Values') && isa(tc.Values,'timeseries')
            data.TotalCost = timeseries(tc.Values.Data, tc.Values.Time);
            fprintf('  ✓ 提取TotalCost: %d数据点\n', length(tc.Values.Data));
        else
            data.TotalCost = tc;
        end
    catch
        % 兼容其他命名
        altNames = {'total_cost','Total_Cost','CostTotal'};
        for iAlt = 1:numel(altNames)
            try
                tc = logsout.getElement(altNames{iAlt});
                data.TotalCost = timeseries(tc.Values.Data, tc.Values.Time);
    % 兼容字段别名：为下游验证提供 P_batt 与 SOH_diff
    if isfield(data,'Battery_Power') && ~isfield(data,'P_batt')
        try
            data.P_batt = data.Battery_Power;
        catch
            % ignore aliasing failure
        end
    end
    if isfield(data,'SOH_Diff') && ~isfield(data,'SOH_diff')
        try
            data.SOH_diff = data.SOH_Diff;
        catch
            % ignore aliasing failure
        end
    end

                fprintf('  ✓ 提取TotalCost(%s): %d数据点\n', altNames{iAlt}, length(tc.Values.Data));
                break;
            catch
            end
        end
    end

    % 可选信号: SOH_Diff (SOH变化量)
    try
        sd = logsout.getElement('SOH_Diff');
        if isprop(sd,'Values') && isa(sd.Values,'timeseries')
            data.SOH_Diff = timeseries(sd.Values.Data, sd.Values.Time);
            fprintf('  ✓ 提取SOH_Diff: %d数据点\n', length(sd.Values.Data));
        else
            data.SOH_Diff = sd;
        end
    catch
        % 兼容其他命名
        altNames = {'SOH_Delta','SOH_Change'};
        for iAlt = 1:numel(altNames)
            try
                sd = logsout.getElement(altNames{iAlt});
                data.SOH_Diff = timeseries(sd.Values.Data, sd.Values.Time);
                fprintf('  ✓ 提取SOH_Diff(%s): %d数据点\n', altNames{iAlt}, length(sd.Values.Data));
                break;
            catch
            end
        end
    end


    % 可选信号: Reward (用于分析) - 支持 'Reward'/'reward'，安全扫描
    try
        rewardTS = [];
        if isa(logsout,'Simulink.SimulationData.Dataset')
            for ii = 1:logsout.numElements
                el = [];
                try
    el = logsout.getElement(ii);
catch
    el = [];
end
                if isempty(el), continue; end
                nm1 = '';
                try
    nm1 = string(el.Name);
catch
    nm1 = '';
end
                if strcmpi(nm1,'reward')
                    if isprop(el,'Values') && isa(el.Values,'timeseries')
                        rewardTS = el.Values; break;
                    elseif isprop(el,'Values') && isa(el.Values,'Simulink.SimulationData.Dataset')
                        val = el.Values;
                        for jj = 1:val.numElements
                            sub = [];
try
    sub = val.getElement(jj);
catch
    sub = [];
end
                            if ~isempty(sub) && isprop(sub,'Values') && isa(sub.Values,'timeseries')
                                rewardTS = sub.Values; break;
                            end
                        end
                        if ~isempty(rewardTS), break; end
                    end
                end
            end
        end
        if ~isempty(rewardTS)
            data.episode_reward = sum(rewardTS.Data, 'omitnan');
            fprintf('  ✓ 提取Reward: 总奖励 = %.2f\n', data.episode_reward);
        end
    catch
        % Reward可选
    end
end

function tf = local_debug_enabled()
    v = '';
    try
        v = getenv('EXTRACT_EPISODE_DEBUG');
    catch
        v = '';
    end
    if isempty(v)
        tf = false;
        return;
    end
    x = str2double(v);
    if isnan(x)
        tf = false;
    else
        tf = (x ~= 0);
    end
end

function data = extract_from_yout(yout, data)
    %EXTRACT_FROM_YOUT 从yout提取信号(旧版Simulink兼容)

    % yout通常是cell array或结构体数组
    % 这里提供基本支持,实际实现需要根据具体模型调整

    fprintf('  ⚠ yout提取功能待完善,请确保使用logsout输出\n');

    % 尝试基本提取
    if iscell(yout) && ~isempty(yout)
        for i = 1:length(yout)
            if isa(yout{i}, 'timeseries')
                ts = yout{i};
                % 根据名称判断是什么信号
                if contains(lower(ts.Name), 'soc')
                    data.Battery_SOC = ts;
                elseif contains(lower(ts.Name), 'soh')
                    data.Battery_SOH = ts;
                elseif contains(lower(ts.Name), 'power') && contains(lower(ts.Name), 'battery')
                    data.Battery_Power = ts;
                end
            end
        end
    end
end

function data = extract_from_workspace(data)
    %EXTRACT_FROM_WORKSPACE 从MATLAB workspace提取变量

    % 尝试从base workspace提取常见变量名
    varNames = {'Battery_SOC','Battery_SOH','Battery_Power','TotalCost','SOH_Diff', ...
                 'battery_soc','battery_soh','total_cost','soh_diff', ...
                 'P_batt','P_Batt','P_battery','P_batt_W','Batt_Power','BatteryPower'};

    for i = 1:length(varNames)
        varName = varNames{i};
        if evalin('base', sprintf('exist(''%s'', ''var'')', varName))
            try
                value = evalin('base', varName);
                % 标准化变量名
                if contains(lower(varName), 'soc')
                    data.Battery_SOC = value;
                    fprintf('  ✓ 从workspace提取: %s\n', varName);
                elseif contains(lower(varName), 'soh')
                    data.Battery_SOH = value;
                    fprintf('  ✓ 从workspace提取: %s\n', varName);
                elseif contains(lower(varName), 'power') || strcmpi(varName,'P_batt') || strcmpi(varName,'P_Batt') ...
                        || strcmpi(varName,'P_battery') || strcmpi(varName,'P_batt_W')
                    data.Battery_Power = value;
                    fprintf('  ✓ 从workspace提取: %s\n', varName);
                end
            catch
                % 忽略提取失败
            end
        end
    end
end

function data = downsample_data(data)
    %DOWNSAMPLE_DATA 降采样时序数据以减小文件大小
    %   原始数据可能是每秒或每分钟采样,降至每小时采样

    fprintf('\n降采样数据以减小文件大小...\n');

    % 处理Battery_SOC
    if isfield(data, 'Battery_SOC') && isa(data.Battery_SOC, 'timeseries')
        originalLength = length(data.Battery_SOC.Data);
        data.Battery_SOC = resample_to_hourly(data.Battery_SOC);
        newLength = length(data.Battery_SOC.Data);
        fprintf('  Battery_SOC: %d → %d 数据点 (降采样%.1f%%)\n', ...
            originalLength, newLength, (1 - newLength/originalLength)*100);
    end

    % 处理Battery_SOH
    if isfield(data, 'Battery_SOH') && isa(data.Battery_SOH, 'timeseries')
        originalLength = length(data.Battery_SOH.Data);
        data.Battery_SOH = resample_to_hourly(data.Battery_SOH);
        newLength = length(data.Battery_SOH.Data);
        fprintf('  Battery_SOH: %d → %d 数据点 (降采样%.1f%%)\n', ...
            originalLength, newLength, (1 - newLength/originalLength)*100);
    end

    % 处理Battery_Power
    if isfield(data, 'Battery_Power') && isa(data.Battery_Power, 'timeseries')
        originalLength = length(data.Battery_Power.Data);
        data.Battery_Power = resample_to_hourly(data.Battery_Power);
        newLength = length(data.Battery_Power.Data);
        fprintf('  Battery_Power: %d → %d 数据点 (降采样%.1f%%)\n', ...
            originalLength, newLength, (1 - newLength/originalLength)*100);
    end


    % 处理TotalCost
    if isfield(data, 'TotalCost') && isa(data.TotalCost, 'timeseries')
        originalLength = length(data.TotalCost.Data);
        data.TotalCost = resample_to_hourly(data.TotalCost);
        newLength = length(data.TotalCost.Data);
        fprintf('  TotalCost: %d → %d 数据点 (降采样%.1f%%)\n', ...
            originalLength, newLength, (1 - newLength/originalLength)*100);
    end

    % 处理SOH_Diff
    if isfield(data, 'SOH_Diff') && isa(data.SOH_Diff, 'timeseries')
        originalLength = length(data.SOH_Diff.Data);
        data.SOH_Diff = resample_to_hourly(data.SOH_Diff);
        newLength = length(data.SOH_Diff.Data);
        fprintf('  SOH_Diff: %d → %d 数据点 (降采样%.1f%%)\n', ...
            originalLength, newLength, (1 - newLength/originalLength)*100);
    end

end

function ts_resampled = resample_to_hourly(ts)
    %RESAMPLE_TO_HOURLY 将时序数据重采样为每小时一个点

    if isempty(ts) || ~isa(ts, 'timeseries')
        ts_resampled = ts;
        return;
    end

    % 获取时间范围
    timeVec = ts.Time;
    if isempty(timeVec)
        ts_resampled = ts;
        return;
    end

    % 计算采样间隔
    dt = median(diff(timeVec));

    % 如果已经是每小时或更稀疏,不需要降采样
    if dt >= 3600  % 已经是每小时或更稀疏
        ts_resampled = ts;
        return;
    end

    % 创建每小时的时间向量
    t_start = timeVec(1);
    t_end = timeVec(end);
    t_hourly = (t_start:3600:t_end)';

    % 重采样
    try
        ts_resampled = resample(ts, t_hourly);
    catch
        % 如果resample失败,使用简单插值
        data_interp = interp1(timeVec, ts.Data, t_hourly, 'linear', 'extrap');
        ts_resampled = timeseries(data_interp, t_hourly);
        ts_resampled.Name = ts.Name;
        ts_resampled.DataInfo.Units = ts.DataInfo.Units;
    end
end

function display_data_summary(data)
    %DISPLAY_DATA_SUMMARY 显示提取的数据统计摘要

    fprintf('\n=== 数据统计摘要 ===\n');

    if isfield(data, 'Battery_SOC') && isa(data.Battery_SOC, 'timeseries')
        soc = data.Battery_SOC.Data;
        if max(soc) <= 1
            soc = soc * 100;  % 转换为百分比显示
        end
        fprintf('  Battery_SOC:\n');
        fprintf('    - 数据点: %d\n', length(soc));
        fprintf('    - 范围: %.1f%% ~ %.1f%%\n', min(soc), max(soc));
    % 若关键字段均缺失，输出明确警告
    keys = {'Battery_SOC','Battery_SOH','Battery_Power','TotalCost','SOH_Diff'};
    hasAny = false;
    for kk = 1:numel(keys)
        if isfield(data, keys{kk})
            hasAny = true; break; end
    end
    if ~hasAny
        fprintf('  ⚠ 未检测到 Battery_SOC/SOH/Power/TotalCost/SOH_Diff 任一字段。\n');
        fprintf('    - 请检查模型参数: SignalLogging=on, SignalLoggingName=logsout (Dataset)\n');
        fprintf('    - 请检查信号线是否勾选 Log signal data 或 To Workspace 是否记为 timeseries\n');
        fprintf('    - 若为 RL 环境，请确保重放时 Fast Restart 关闭（本函数已尝试）\n');
    end

        fprintf('    - 平均值: %.1f%%\n', mean(soc));
    end

    if isfield(data, 'Battery_SOH') && isa(data.Battery_SOH, 'timeseries')
        soh = data.Battery_SOH.Data;
        if max(soh) <= 1
            soh = soh * 100;
        end
        fprintf('  Battery_SOH:\n');
        fprintf('    - 数据点: %d\n', length(soh));
        fprintf('    - 范围: %.1f%% ~ %.1f%%\n', min(soh), max(soh));
        fprintf('    - 衰减: %.2f%%\n', 100 - min(soh));
    end

    if isfield(data, 'TotalCost')
        if isa(data.TotalCost, 'timeseries')
            tc = data.TotalCost.Data;
            fprintf('  TotalCost:\n');
            fprintf('    - 数据点: %d\n', length(tc));
            fprintf('    - 总成本: %.2f\n', sum(tc,'omitnan'));
        elseif isnumeric(data.TotalCost)
            fprintf('  TotalCost: %.2f\n', double(data.TotalCost));
        end
    end

    if isfield(data, 'SOH_Diff')
        if isa(data.SOH_Diff, 'timeseries')
            sd = data.SOH_Diff.Data;
            fprintf('  SOH_Diff:\n');
            fprintf('    - 数据点: %d\n', length(sd));
            fprintf('    - 总变化: %.4f\n', sum(sd,'omitnan'));
        elseif isnumeric(data.SOH_Diff)
            fprintf('  SOH_Diff: %.4f\n', double(data.SOH_Diff));
        end
    end


    if isfield(data, 'Battery_Power') && isa(data.Battery_Power, 'timeseries')
        powW = double(data.Battery_Power.Data);
        fprintf('  Battery_Power:\n');
        fprintf('    - 数据点: %d\n', length(powW));
        fprintf('    - 原始范围: %.1f ~ %.1f (原始单位)\n', min(powW), max(powW));
        % 额外打印 kW 范围，便于人工判断单位
        powkW = powW / 1000;
        fprintf('    - 换算范围: %.2f kW ~ %.2f kW\n', min(powkW), max(powkW));
    end

    if isfield(data, 'cumulative_cost')
        fprintf('  经济指标:\n');
        fprintf('    - 累计成本: $%.2f\n', data.cumulative_cost);
    end

    if isfield(data, 'episode_reward')
        fprintf('  奖励指标:\n');
        fprintf('    - 总奖励: %.2f\n', data.episode_reward);
    end

    fprintf('=====================\n');
end



function data = extract_from_simout_clean(simOut)
%EXTRACT_FROM_SIMOUT_CLEAN 从 simOut 结构体提取关键信号（带调试输出）

    data = struct();

    % :  simOut/logsout 
    try
        fprintf('DEBUG: simOut : %s\n', class(simOut));
        if isfield(simOut,'logsout')
            ds = simOut.logsout;
            fprintf('DEBUG: logsout : %s\n', class(ds));
            if isa(ds,'Simulink.SimulationData.Dataset')
                fprintf('DEBUG: logsout 元素数: %d\n', ds.numElements);
                for ii = 1:ds.numElements
                    elt = [];
                    try
                        elt = ds.getElement(ii);
                    catch

                        try
                            elt = ds.get(ii);
                        catch
                        end
                    end
                    if ~isempty(elt)
                        try
                            fprintf('  - %s\n', elt.Name);
                        catch
                        end
                    end
                end
            end
        else
            fprintf('DEBUG: simOut  logsout \n');
        end
    catch
    end

    % logsout
    if isfield(simOut, 'logsout')
        data = extract_from_logsout(simOut.logsout, data);
    end

    % yout
    if isfield(simOut, 'yout') && isempty(fieldnames(data))
        fprintf('   logsout, yout...\n');
        data = extract_from_yout(simOut.yout, data);
    end

    % workspace fallback
    if isempty(fieldnames(data))

        fprintf('   simOut, workspace...\n');
        data = extract_from_workspace(data);
    end

    % downsample
    if ~isempty(fieldnames(data))
        data = downsample_data(data);
    end
end


function data = extract_from_simout_clean_v2(simOut)
%EXTRACT_FROM_SIMOUT_CLEAN_V2 从 simOut 提取关键信号（兼容 RL 仿真返回）

    data = struct();

    % 定位 logsout（兼容多种返回格式）
    logsout = [];
    % 1) 优先使用 SimulationOutput 的 get 接口（支持对象动态属性）
    try
        logsout = simOut.get('logsout');
    catch
    end
    % 2) 直接点取属性（若 get 不支持/未创建则可能抛异常）
    if isempty(logsout)
        try
            logsout = simOut.logsout;
        catch
        end
    end
    % 3) 兼容 RL 仿真返回格式
    if isempty(logsout)
        try
            logsout = simOut.SimulationInfo.SimulationOutput.logsout;
        catch
        end
    end
    % 4) 其他包装形式
    if isempty(logsout)
        try
            logsout = simOut.simout.logsout;
        catch
        end
    end
    % 5) 最后尝试 get(simOut) 返回的结构体中探测
    if isempty(logsout)
        try
            sAll = get(simOut);
            if isstruct(sAll) && isfield(sAll,'logsout')
                logsout = sAll.logsout;
            end
        catch
        end
    end

    % 调试打印
    try
        fprintf('DEBUG: simOut 类型: %s\n', class(simOut));
        if ~isempty(logsout)
            fprintf('DEBUG: logsout 类型: %s\n', class(logsout));
            if isa(logsout,'Simulink.SimulationData.Dataset')
                fprintf('DEBUG: logsout 元素数: %d\n', logsout.numElements);
                for ii = 1:logsout.numElements
                    el = [];
                    try
                        el = logsout.getElement(ii);
                    catch
                        try
                            el = logsout.get(ii);
                        catch
                            el = [];
                        end
                    end
                    if ~isempty(el)
                        try
                            fprintf('  - %s\n', el.Name);
                        catch
                        end
                    end
                end
            end
        else
            fprintf('DEBUG: 未找到 logsout 字段\n');
        end
    catch
    end

    % 提取
    if ~isempty(logsout)
        data = extract_from_logsout(logsout, data);
    end

    if isempty(fieldnames(data)) && isfield(simOut,'yout')
        fprintf('  ⚠ logsout 为空，尝试从 yout 提取...\n');
        data = extract_from_yout(simOut.yout, data);
    end

    if isempty(fieldnames(data))
        fprintf('  ⚠ simOut 未含目标信号，尝试从 workspace 提取...\n');
        data = extract_from_workspace(data);
    end

    if ~isempty(fieldnames(data))
        data = downsample_data(data);
    end
end
