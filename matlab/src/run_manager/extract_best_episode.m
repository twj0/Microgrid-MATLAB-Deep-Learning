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

    episodeData = struct();

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

        simOpts = rlSimulationOptions('MaxSteps', maxSteps, 'UseParallel', false);
        simOut = sim(env, agent, simOpts);
        episodeData = extract_from_simout_clean_v2(simOut);

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
    if false

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
                try, fprintf('    - %s\n', e.Name); catch, end
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
        for iN = 1:numel(socNames)
            try
                soc = logsout.getElement(socNames{iN});
                break;
            catch
            end
        end
        if isempty(soc)
            % fallback: scan elements for name containing 'soc'
            if isa(logsout,'Simulink.SimulationData.Dataset')
                for ii = 1:logsout.numElements
                    el = [];
                    try
                        el = logsout.getElement(ii);
                    catch
                        el = [];
                    end
                    if ~isempty(el)
                        try
                            nm = lower(string(el.Name));
                            if contains(nm,'soc')
                                soc = el; break;
                            end
                        catch, end
                    end
                end
            end
        end
        if ~isempty(soc) && isprop(soc,'Values') && isa(soc.Values,'timeseries')
            data.Battery_SOC = timeseries(soc.Values.Data, soc.Values.Time);
            fprintf('  ✓ 提取Battery_SOC: %d数据点\n', length(soc.Values.Data));
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
        for iN = 1:numel(sohNames)
            try
                soh = logsout.getElement(sohNames{iN});
                break;
            catch
            end
        end
        if isempty(soh)
            if isa(logsout,'Simulink.SimulationData.Dataset')
                for ii = 1:logsout.numElements
                    el = [];
                    try
                        el = logsout.getElement(ii);
                    catch
                        el = [];
                    end
                    if ~isempty(el)
                        try
                            nm = lower(string(el.Name));
                            if contains(nm,'soh')
                                soh = el; break;
                            end
                        catch, end
                    end
                end
            end
        end
        if ~isempty(soh) && isprop(soh,'Values') && isa(soh.Values,'timeseries')
            data.Battery_SOH = timeseries(soh.Values.Data, soh.Values.Time);
            fprintf('  ✓ 提取Battery_SOH: %d数据点\n', length(soh.Values.Data));
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
                power = logsout.getElement(pwrNames{iN});
                break;
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
                if ~isempty(el)
                    try
                        nm = lower(string(el.Name));
                        if (contains(nm,'batt') || contains(nm,'battery')) && contains(nm,'power')
                            power = el; break;
                        end
                    catch, end
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

    % 可选信号: Grid_Cost (用于经济分析)
    try
        cost = logsout.getElement('Grid_Cost');
        cost_data = cost.Values.Data;
        data.cumulative_cost = sum(cost_data);
        data.average_hourly_cost = mean(cost_data);
        fprintf('  ✓ 提取Grid_Cost: 累计成本 = %.2f\n', data.cumulative_cost);
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


    % 可选信号: Reward (用于分析)
    try
        reward = logsout.getElement('Reward');
        data.episode_reward = sum(reward.Values.Data);
        fprintf('  ✓ 提取Reward: 总奖励 = %.2f\n', data.episode_reward);
    catch
        % Reward可选
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
        power = data.Battery_Power.Data / 1000;  % 转换为kW
        fprintf('  Battery_Power:\n');
        fprintf('    - 数据点: %d\n', length(power));
        fprintf('    - 范围: %.1f kW ~ %.1f kW\n', min(power), max(power));
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
    try
        if isfield(simOut,'logsout')
            logsout = simOut.logsout;
        elseif isfield(simOut,'SimulationInfo')
            try
                logsout = simOut.SimulationInfo.SimulationOutput.logsout;
            catch
            end
        elseif isfield(simOut,'simout')
            try
                logsout = simOut.simout.logsout;
            catch
            end
        end
    catch
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
