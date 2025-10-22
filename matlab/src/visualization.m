% utf-8
function visualization(options)
%VISUALIZATION Generate SAC training and microgrid strategy figures.
%   visualization(options) creates a collection of publication-ready plots
%   that summarise Soft Actor-Critic training progress and microgrid
%   operating strategies. The function searches the specified workspace for
%   training statistics, environment profiles, and economic reward
%   components, then saves annotated figures to the configured results
%   directory.
%
%   Options (all optional):
%       workspace       : 'base' (default), 'caller', or struct container
%       saveFigures     : logical, default true
%       showFigures     : logical, default false (figures rendered off-screen)
%       outputDir       : destination folder (auto-created when needed)
%       filePrefix      : prefix for saved figure files (default 'SAC')
%       figureFormat    : image format extension, default 'png'
%       closeAfterSave  : logical, default true
%       timestamp       : custom timestamp string for filenames
%       trainingResults : struct with aggregated training metrics
%       trainingStats   : raw rlTrainingResult structure (EpisodeReward etc.)
%       pv_power_profile, load_power_profile, price_profile : input data

    if nargin < 1 || isempty(options)
        options = struct();
    end

    opts.workspace      = char(get_option(options, 'workspace', 'base'));
    opts.saveFigures    = logical(get_option(options, 'saveFigures', true));
    opts.showFigures    = logical(get_option(options, 'showFigures', false));
    opts.outputDir      = char(get_option(options, 'outputDir', fullfile(pwd, '..', 'results')));
    opts.filePrefix     = char(get_option(options, 'filePrefix', 'SAC'));
    opts.figureFormat   = lower(char(get_option(options, 'figureFormat', 'png')));
    opts.closeAfterSave = logical(get_option(options, 'closeAfterSave', true));
    opts.timestamp      = char(get_option(options, 'timestamp', char(datetime('now','Format','yyyyMMdd_HHmmss'))));


    % Initialize publication-style defaults
    viz_init_style();

    if opts.saveFigures && ~exist(opts.outputDir, 'dir')
        mkdir(opts.outputDir);
    end

    fprintf('\n=== SAC 可视化分析 ===\n');
    fprintf('  • 输出目录: %s\n', opts.outputDir);
    fprintf('  • 时间戳  : %s\n', opts.timestamp);

    trainingResults = get_option(options, 'trainingResults', []);
    trainingStats   = get_option(options, 'trainingStats', []);
    pvProfile       = get_option(options, 'pv_power_profile', []);
    loadProfile     = get_option(options, 'load_power_profile', []);
    priceProfile    = get_option(options, 'price_profile', []);
    rewardMonitor   = get_option(options, 'rewardMonitor', []);

    if isempty(trainingResults)
        trainingResults = fetch_workspace_variable(opts.workspace, ...
            ["sac_training_results", "training_results", "trainingSummary", "results"]);
    end
    if isempty(trainingStats)
        trainingStats = fetch_workspace_variable(opts.workspace, ...
            ["sac_training_stats", "trainingStats", "training_history", "stats"]);
    end
    if isempty(pvProfile)
        pvProfile = fetch_workspace_variable(opts.workspace, ["pv_power_profile", "pvProfile"]);
    end
    if isempty(loadProfile)
        loadProfile = fetch_workspace_variable(opts.workspace, ["load_power_profile", "loadProfile"]);
    end
    if isempty(priceProfile)
        priceProfile = fetch_workspace_variable(opts.workspace, ["price_profile", "priceProfile"]);
    end

    if isempty(rewardMonitor)
        rewardMonitor = collect_field(trainingResults, 'reward_monitor');
    end
    if isempty(rewardMonitor)
        rewardMonitor = fetch_workspace_variable(opts.workspace, ["dqn_reward_monitor", "reward_monitor", "rewardMonitor"]);
    end

    rewardSeries = collect_field(trainingResults, 'episode_rewards');
    if isempty(rewardSeries)
        rewardSeries = collect_field(trainingStats, 'EpisodeReward');
    end
    episodeIndices = (1:numel(rewardSeries))';

    averageRewardSeries = collect_field(trainingStats, 'AverageReward');
    episodeStepsSeries  = collect_field(trainingStats, 'EpisodeSteps');
    q0Series            = collect_field(trainingStats, 'EpisodeQ0');

    pvData = struct();
    [pvData.timeHours, pvData.power] = timeseries_to_array(pvProfile);
    [loadData.timeHours, loadData.power] = timeseries_to_array(loadProfile);
    [priceData.timeHours, priceData.price] = timeseries_to_array(priceProfile);

    [strategy, strategyMeta] = derive_strategy(pvData, loadData, priceData);

    figures = {};
    figures{end+1} = create_training_overview(opts, rewardSeries, episodeIndices, ...
        averageRewardSeries, episodeStepsSeries, q0Series, trainingResults);
    figures{end+1} = create_reward_monitor(opts, rewardMonitor, 2000);
    figures{end+1} = create_microgrid_profiles(opts, pvData, loadData, priceData, strategy);
    figures{end+1} = create_strategy_summary(opts, rewardSeries, trainingResults, strategyMeta, ...
        pvData, loadData, priceData);

    % Create additional scientific visualization components for research publication
    fprintf('\n=== 科研级可视化分析 ===\n');

    % 4. Power flow analysis visualization
    try
        figures{end+1} = create_power_flow_analysis(opts, pvData, loadData, priceData);
        fprintf('  ✓ 功率流分析可视化完成\n');
    catch ME
        fprintf('  ⚠ 功率流分析失败: %s\n', ME.message);
    end

    % 5. Economic analysis visualization
    try
        figures{end+1} = create_economic_analysis(opts, pvData, loadData, priceData);
        fprintf('  ✓ 经济分析可视化完成\n');
    catch ME
        fprintf('  ⚠ 经济分析失败: %s\n', ME.message);
    end

    % 6. System performance visualization
    try
        figures{end+1} = create_system_performance(opts, pvData, loadData, priceData);
        fprintf('  ✓ 系统性能可视化完成\n');
    catch ME
        fprintf('  ⚠ 系统性能分析失败: %s\n', ME.message);
    end

    % 7. Battery performance visualization (SOC/SOH)
    try
        batterySOC = fetch_workspace_variable(opts.workspace, ["Battery_SOC", "battery_soc", "soc", "SOC"]);
        batterySOH = fetch_workspace_variable(opts.workspace, ["Battery_SOH", "battery_soh", "soh", "SOH"]);
        batteryPower = fetch_workspace_variable(opts.workspace, ["Battery_Power", "battery_power", "batteryPower"]);
        batterySOHDiff = fetch_workspace_variable(opts.workspace, ["SOH_Diff", "battery_soh_diff", "soh_diff", "SOH_Diff"]);
        hasBatteryData = ~isempty(batterySOC) || ~isempty(batterySOH) || ~isempty(batterySOHDiff);
        if hasBatteryData
            figures{end+1} = create_battery_performance(opts, batterySOC, batterySOH, batteryPower);
            figures{end+1} = create_battery_summary(opts, batterySOC, batterySOH, batterySOHDiff);
            fprintf('  ✓ 电池性能可视化完成\n');
        else
            fprintf('  ⊘ 未检测到电池数据(Battery_SOC/Battery_SOH),跳过电池可视化\n');
        end
    catch ME
        fprintf('  ⚠ 电池性能分析失败: %s\n', ME.message);
    end
    % === P1.5 论文核心图 (Fig A–D) ===
    try
        % 统一到每小时timetable
        tt = preprocess_timeseries(pvProfile, loadProfile, priceProfile);

        % 尝试注入 Battery_Power 与 Battery_SOC
        batteryPowerVar = fetch_workspace_variable(opts.workspace, ["Battery_Power", "battery_power", "batteryPower"]);
        batterySOCVar   = fetch_workspace_variable(opts.workspace, ["Battery_SOC", "battery_soc", "soc", "SOC"]);
        % 若工作区未找到，尝试从 results/best_run/best_episode_data.mat 读取
        if (isempty(batteryPowerVar) || isempty(batterySOCVar))
            try
                thisFile = mfilename('fullpath');
                projRoot = fileparts(fileparts(thisFile)); % .../matlab/src -> 项目根目录
                dataMat = fullfile(projRoot, 'results', 'best_run', 'best_episode_data.mat');
                if isfile(dataMat)
                    S = load(dataMat);
                    if isempty(batteryPowerVar) && isfield(S, 'Battery_Power')
                        batteryPowerVar = S.Battery_Power;
                    end
                    if isempty(batterySOCVar) && isfield(S, 'Battery_SOC')
                        batterySOCVar = S.Battery_SOC;
                    end
                end
            catch
                % 忽略读取失败，后续按缺省处理
            end
        end

        baseT0 = datetime(2000,1,1,0,0,0);
        if ~isempty(batteryPowerVar)
            [tb, pb] = timeseries_to_array(batteryPowerVar);
            ttBatt = timetable(baseT0 + hours(tb), double(pb(:)), 'VariableNames', {'Battery_Power'});
            tt = synchronize(tt, ttBatt, 'union', 'mean');
            tt = retime(tt, 'hourly', 'mean');
            if any(ismissing(tt.Battery_Power))
                tt.Battery_Power = fillmissing(tt.Battery_Power, 'constant', 0);
            end
        else
            tt.Battery_Power = zeros(height(tt),1);
        end
        if ~isempty(batterySOCVar)
            [tsoc, soc] = timeseries_to_array(batterySOCVar);
            ttSOC = timetable(baseT0 + hours(tsoc), double(soc(:)), 'VariableNames', {'Battery_SOC'});
            tt = synchronize(tt, ttSOC, 'union', 'mean');
            tt = retime(tt, 'hourly', 'mean');
            tt.Battery_SOC = fillmissing(tt.Battery_SOC, 'linear', 'EndValues','nearest');
        end

        % 计算并网购电功率(Grid_Power)，无售电: >=0
        tt.Grid_Power = compute_grid_power(tt.PV_Power, tt.Load_Power, tt.Battery_Power);

        % 生成 Fig A–D
        figA = plot_fig_A_overview(opts, tt, []); if isgraphics(figA), figA.UserData = struct('label','figA_overview'); figures{end+1} = figA; end
        figB = plot_fig_B_battery(opts, tt);      if isgraphics(figB), figB.UserData = struct('label','figB_battery');   figures{end+1} = figB; end
        figC = plot_fig_C_grid_exchange(opts, tt);if isgraphics(figC), figC.UserData = struct('label','figC_grid');      figures{end+1} = figC; end
        figD = plot_fig_D_supply_stack(opts, tt); if isgraphics(figD), figD.UserData = struct('label','figD_supply');    figures{end+1} = figD; end
        fprintf('  ✓ Fig A–D 论文图型生成完成\n');
    catch ME
        fprintf('  ⚠ 论文图型(Fig A–D)生成失败: %s\n', ME.message);
    end


    savedCount = save_figures(figures, opts);
    fprintf('  ✓ 图像生成完成 (共%d个)\n\n', savedCount);
end

function fig = create_training_overview(opts, rewards, episodes, avgReward, episodeSteps, q0Series, trainingResults)
    % 根据调用方的filePrefix自适应标题，避免误标算法名称
    fig = make_figure(sprintf('%s训练总览', char(get_option(opts, 'filePrefix', 'SAC'))), opts.showFigures, [100, 100, 1400, 800]);
    tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    if isempty(rewards)
        text(0.5, 0.5, '无训练奖励数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
    else
        plot(episodes, rewards, 'Color', [0.3, 0.5, 0.9], 'LineWidth', 1.3);
        hold on;
        if numel(rewards) >= 10
            window = max(10, ceil(numel(rewards) / 10));
            smoothReward = movmean(rewards, window);
            plot(episodes, smoothReward, 'r-', 'LineWidth', 1.8);
            legend({'Episode Reward', sprintf('%d-ep Moving Avg', window)}, 'Location', 'best');
        else
            legend({'Episode Reward'}, 'Location', 'best');
        end
        hold off;
        xlabel('Episode');
        ylabel('Reward');
        title('Episode Reward 轨迹');
        grid on;
    end

    nexttile;
    if isempty(episodeSteps)
        text(0.5, 0.5, '无Episode步数数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
    else
        bar(episodes, episodeSteps, 'FaceColor', [0.2, 0.6, 0.6], 'EdgeColor', 'none');
        xlabel('Episode');
        ylabel('Steps');
        title('每回合仿真步数');
        grid on;
    end

    nexttile;
    if isempty(rewards)
        text(0.5, 0.5, '无奖励分布数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
    else
        histogram(rewards, 'FaceColor', [0.4, 0.4, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
        xlabel('Reward');
        ylabel('Frequency');
        title('奖励分布');
        grid on;
    end

    nexttile;
    if ~isempty(q0Series)
        plot(episodes, q0Series, 'm-', 'LineWidth', 1.4);
        xlabel('Episode');
        ylabel('Q0 Value');
        title('Q0收敛趋势');
        grid on;
    elseif ~isempty(avgReward)
        plot(episodes(1:numel(avgReward)), avgReward, 'g-', 'LineWidth', 1.6);
        xlabel('Episode');
        ylabel('Average Reward');
        title('平均奖励走势');
        grid on;
    elseif ~isempty(rewards)
        cumAvg = cumsum(rewards) ./ (1:numel(rewards))';
        plot(episodes, cumAvg, 'g-', 'LineWidth', 1.6);
        xlabel('Episode');
        ylabel('Cumulative Avg');
        title('奖励累计平均');
        grid on;
    else
        text(0.5, 0.5, '无可用数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
    end

    add_training_annotation(fig, rewards, trainingResults);
end

function fig = create_reward_monitor(opts, rewardMonitor, targetEpisodes)
    if nargin < 3 || isempty(targetEpisodes)
        targetEpisodes = 2000;
    end
    fig = make_figure('DQN奖励监控', opts.showFigures, [120, 120, 1400, 600]);
    if isempty(rewardMonitor)
        axis off;
        text(0.5, 0.5, '未找到奖励监控数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
        return;
    end

    if isstruct(rewardMonitor)
        episodes = to_column(get_struct_field(rewardMonitor, 'episodes'));
        modelReward = to_column(get_struct_field(rewardMonitor, 'model_reward'));
        matlabReward = to_column(get_struct_field(rewardMonitor, 'matlab_reward'));
        totalReward = to_column(get_struct_field(rewardMonitor, 'total_reward'));
        averageReward = to_column(get_struct_field(rewardMonitor, 'average_reward'));
    else
        episodes = to_column(get_option(rewardMonitor, 'episodes', []));
        modelReward = to_column(get_option(rewardMonitor, 'model_reward', []));
        matlabReward = to_column(get_option(rewardMonitor, 'matlab_reward', []));
        totalReward = to_column(get_option(rewardMonitor, 'total_reward', []));
        averageReward = to_column(get_option(rewardMonitor, 'average_reward', []));
    end

    if isempty(episodes)
        episodes = (1:max([numel(modelReward), numel(totalReward), numel(matlabReward), numel(averageReward), targetEpisodes]))';
    end

    dataLength = numel(episodes);
    idxMask = episodes <= targetEpisodes;
    if any(idxMask)
        episodes = episodes(idxMask);
        modelReward = mask_and_trim(modelReward, idxMask);
        matlabReward = mask_and_trim(matlabReward, idxMask);
        totalReward = mask_and_trim(totalReward, idxMask);
        averageReward = mask_and_trim(averageReward, idxMask);
    else
        episodes = episodes(1:min(dataLength, targetEpisodes));
        modelReward = resize_and_trim(modelReward, numel(episodes));
        matlabReward = resize_and_trim(matlabReward, numel(episodes));
        totalReward = resize_and_trim(totalReward, numel(episodes));
        averageReward = resize_and_trim(averageReward, numel(episodes));
    end

    if isempty(totalReward) && ~isempty(modelReward) && ~isempty(matlabReward)
        totalReward = modelReward + matlabReward;
    end
    if isempty(matlabReward) && ~isempty(totalReward) && ~isempty(modelReward)
        matlabReward = totalReward - modelReward;
    end
    if isempty(modelReward) && ~isempty(totalReward) && ~isempty(matlabReward)
        modelReward = totalReward - matlabReward;
    end
    if isempty(averageReward) && ~isempty(totalReward)
        averageReward = cumsum(totalReward) ./ (1:numel(totalReward))';
    end

    hold on;
    plot_series(episodes, matlabReward, [0.30 0.75 0.93], 1.2, 'MATLAB奖励');
    plot_series(episodes, modelReward, [0.00 0.45 0.74], 1.5, '模型奖励');
    plot_series(episodes, totalReward, [0.85 0.33 0.10], 1.8, '总奖励');
    plot_series(episodes, averageReward, [0.47 0.67 0.19], 2.0, '平均总奖励');
    hold off;

    xlabel('Episode');
    ylabel('绘画奖励');
    title(sprintf('奖励监控（前%d个Episode）', targetEpisodes));
    legend('Location', 'best');
    grid on;
    xlim([1, max(targetEpisodes, max(episodes))]);

    if opts.saveFigures
        set(gca, 'FontName', 'Arial', 'FontSize', 12);
    end

    function col = to_column(value)
        if isempty(value)
            col = [];
        else
            col = double(value(:));
        end
    end

    function value = get_struct_field(s, fieldName)
        if isfield(s, fieldName)
            value = s.(fieldName);
        else
            value = [];
        end
    end

    function out = mask_and_trim(vec, mask)
        if isempty(vec)
            out = [];
        else
            vec = to_column(vec);
            mask = mask(1:min(numel(mask), numel(vec)));
            out = vec(1:numel(mask));
            out = out(mask);
        end
    end

    function out = resize_and_trim(vec, newLength)
        if isempty(vec)
            out = [];
        else
            vec = to_column(vec);
            out = vec(1:min(numel(vec), newLength));
        end
    end

    function plot_series(x, y, colorValue, widthValue, labelValue)
        if isempty(y)
            return;
        end
        plot(x(1:numel(y)), y, 'LineWidth', widthValue, 'Color', colorValue, 'DisplayName', labelValue);
    end
end

function fig = create_microgrid_profiles(opts, pvData, loadData, priceData, strategy)
    fig = make_figure('微电网运行概览', opts.showFigures, [150, 150, 1400, 900]);
    tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    if isempty(pvData.power) || isempty(loadData.power)
        text(0.5, 0.5, '缺少光伏/负载曲线数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
    else
        [pvTime, pvPower] = align_and_clip_series(pvData.timeHours, pvData.power, 72);
        [loadTime, loadPower] = align_and_clip_series(loadData.timeHours, loadData.power, 72);
        netLoad = loadPower - pvPower;

        plot(pvTime, pvPower/1000, 'Color', [1.0, 0.65, 0.0], 'LineWidth', 1.5);
        hold on;
        plot(loadTime, loadPower/1000, 'Color', [0.2, 0.3, 0.8], 'LineWidth', 1.5);
        plot(loadTime, netLoad/1000, '--', 'Color', [0.4, 0.7, 0.4], 'LineWidth', 1.2);
        hold off;
        ylabel('功率 (kW)');
        xlabel('时间 (小时)');
        title('前72小时光伏/负载功率');
        legend({'PV Output', 'Load', 'Net Load'}, 'Location', 'best');
        grid on;
    end

    nexttile;
    if isempty(priceData.price)
        text(0.5, 0.5, '缺少电价数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
    else
        [priceTime, priceSeries] = align_and_clip_series(priceData.timeHours, priceData.price, 72);
        plot(priceTime, priceSeries, 'Color', [0.85, 0.33, 0.1], 'LineWidth', 1.5);
        xlabel('时间 (小时)');
        ylabel('电价 ($/kWh)');
        title('电价曲线 (前72小时)');
        grid on;
    end

    nexttile;
    if isempty(strategy.timeHours) || isempty(strategy.bestActionKW)
        text(0.5, 0.5, '策略建议数据不可用', 'HorizontalAlignment', 'center', 'FontSize', 12);
    else
        plot(strategy.timeHours, strategy.bestActionKW, 'LineWidth', 1.5, 'Color', [0.2, 0.6, 0.6]);
        hold on;
        yline(0, 'k--', 'LineWidth', 1.0);
        hold off;
        xlabel('时间 (小时)');
        ylabel('电池功率 (kW)');
        title('基于经济奖励的策略建议 (正充负放)');
        grid on;
    end
end

function fig = create_strategy_summary(opts, rewards, trainingResults, strategyMeta, pvData, loadData, priceData)
    fig = make_figure('训练与策略总结', opts.showFigures, [180, 180, 1100, 840]);
    axis off;

    summaryLines = compose_summary_text(rewards, trainingResults, strategyMeta, pvData, loadData, priceData);
    text(0.05, 0.95, summaryLines, ...
        'FontName', 'Courier New', ...
        'FontSize', 10, ...
        'VerticalAlignment', 'top', ...
        'Interpreter', 'none');
end

function add_training_annotation(figHandle, rewards, trainingResults)
    if isempty(rewards) || ~ishandle(figHandle)
        return;
    end

    totalEpisodes = numel(rewards);
    window = min(10, totalEpisodes);
    initialAvg = mean(rewards(1:window));
    finalAvg = mean(rewards(max(1, totalEpisodes - window + 1):end));
    bestReward = max(rewards);
    worstReward = min(rewards);

    trainingTime = collect_field(trainingResults, 'training_time');
    if isempty(trainingTime)
        formattedTime = '未知';
    else
        formattedTime = sprintf('%.1f 分钟', trainingTime / 60);
    end

    improvement = finalAvg - initialAvg;
    annotation(figHandle, 'textbox', [0.55, 0.52, 0.4, 0.4], ...
        'String', { ...
            sprintf('总回合数       : %d', totalEpisodes), ...
            sprintf('初始10回合均值 : %.2f', initialAvg), ...
            sprintf('末尾10回合均值 : %.2f', finalAvg), ...
            sprintf('累计提升       : %.2f', improvement), ...
            sprintf('最佳奖励       : %.2f', bestReward), ...
            sprintf('最差奖励       : %.2f', worstReward), ...
            sprintf('训练用时       : %s', formattedTime) ...
        }, ...
        'FontSize', 10, ...
        'BackgroundColor', [1, 1, 1, 0.85], ...
        'EdgeColor', [0.8, 0.8, 0.8], ...
        'FitBoxToText', 'on');
end

function [seriesTime, seriesData] = timeseries_to_array(ts)
    seriesTime = [];
    seriesData = [];
    if isempty(ts)
        return;
    end

    if isa(ts, 'timeseries')
        [seriesTime, seriesData] = handle_timeseries_input(ts);
    elseif istimetable(ts) || istable(ts)
        [seriesTime, seriesData] = table_like_to_arrays(ts);
    elseif isstruct(ts)
        [seriesTime, seriesData] = struct_to_series(ts);
    else
        seriesData = ensure_numeric_column(ts);
        if isempty(seriesData)
            return;
        end
    end

    if isempty(seriesTime) && ~isempty(seriesData)
        seriesTime = (0:numel(seriesData)-1)';
    end

    if numel(seriesTime) > numel(seriesData)
        seriesTime = seriesTime(1:numel(seriesData));
    elseif numel(seriesData) > numel(seriesTime)
        seriesData = seriesData(1:numel(seriesTime));
    end
end

function [seriesTime, seriesData] = handle_timeseries_input(ts)
    rawTime = ts.Time;
    seriesTime = normalize_time_vector(rawTime);

    data = double(ts.Data);
    if ts.IsTimeFirst
        numSamples = size(data, 1);
        reshaped = reshape(data, numSamples, []);
    else
        permOrder = [2, 1, 3:ndims(data)];
        dataPerm = permute(data, permOrder);
        numSamples = size(dataPerm, 1);
        reshaped = reshape(dataPerm, numSamples, []);
    end
    seriesData = ensure_numeric_column(reshaped(:, 1));
end

function [seriesTime, seriesData] = table_like_to_arrays(tbl)
    seriesTime = [];
    seriesData = [];
    if isempty(tbl)
        return;
    end

    if istimetable(tbl)
        seriesTime = normalize_time_vector(tbl.Properties.RowTimes);
        varNames = tbl.Properties.VariableNames;
        for idx = 1:numel(varNames)
            candidate = ensure_numeric_column(tbl{:, varNames{idx}});
            if ~isempty(candidate)
                seriesData = candidate;
                return;
            end
        end
        return;
    end

    varNames = tbl.Properties.VariableNames;
    timeVar = '';
    for idx = 1:numel(varNames)
        name = varNames{idx};
        lowerName = lower(char(name));
        if contains(lowerName, 'time') || contains(lowerName, 'hour')
            seriesTime = normalize_time_vector(tbl.(name), name);
            timeVar = char(name);
            break;
        end
    end

    for idx = 1:numel(varNames)
        name = varNames{idx};
        if ~isempty(timeVar) && strcmp(char(name), timeVar)
            continue;
        end
        candidate = ensure_numeric_column(tbl.(name));
        if ~isempty(candidate)
            seriesData = candidate;
            return;
        end
    end

    try
        seriesData = ensure_numeric_column(table2array(tbl));
    catch
        seriesData = [];
    end
end

function [seriesTime, seriesData] = struct_to_series(s)
    seriesTime = [];
    seriesData = [];
    if isempty(s)
        return;
    end

    if numel(s) > 1
        try
            tbl = struct2table(s);
            [seriesTime, seriesData] = table_like_to_arrays(tbl);
            if ~isempty(seriesData)
                return;
            end
        catch
        end
    end

    fields = fieldnames(s);
    if isempty(fields)
        return;
    end

    timeField = detect_struct_field(fields, true);
    dataField = detect_struct_field(fields, false);

    if ~isempty(dataField) && isfield(s, dataField)
        seriesData = ensure_numeric_column(s.(dataField));
    end
    if ~isempty(timeField) && isfield(s, timeField)
        seriesTime = normalize_time_vector(s.(timeField), timeField);
    end

    if isempty(seriesData)
        for idx = 1:numel(fields)
            name = fields{idx};
            if strcmp(name, timeField)
                continue;
            end
            candidate = ensure_numeric_column(s.(name));
            if ~isempty(candidate)
                seriesData = candidate;
                break;
            end
        end
    end
end

function field = detect_struct_field(fieldNames, isTimeField)
    field = '';
    if isempty(fieldNames)
        return;
    end
    lowerFields = lower(string(fieldNames));

    if isTimeField
        patterns = ["timehours", "time_hours", "timehour", "time", "timestamp", "datetime", "hours", "hour", "rowtime"];
        skipTokens = ["info", "meta", "label", "unit"];
    else
        patterns = ["power", "load", "pv", "price", "data", "values", "value", "series", "signal", "measure"];
        skipTokens = ["info", "meta", "label", "unit", "time"];
    end

    for patIdx = 1:numel(patterns)
        matches = find(contains(lowerFields, patterns(patIdx)));
        for m = matches(:)'
            candidate = fieldNames{m};
            candidateLower = lower(candidate);
            if any(contains(candidateLower, skipTokens))
                continue;
            end
            field = candidate;
            return;
        end
    end

    if isTimeField
        altMatches = find(ismember(lowerFields, ["t", "x"]), 1);
        if ~isempty(altMatches)
            field = fieldNames{altMatches};
        end
    else
        for idx = 1:numel(fieldNames)
            candidateLower = lower(fieldNames{idx});
            if contains(candidateLower, ["time", "hour", "date", "meta", "info", "label", "unit"])
                continue;
            end
            field = fieldNames{idx};
            return;
        end
    end
end

function column = ensure_numeric_column(value)
    column = [];
    if isempty(value)
        return;
    end

    if isa(value, 'timeseries')
        [~, column] = handle_timeseries_input(value);
        return;
    end
    if istimetable(value) || istable(value)
        [~, column] = table_like_to_arrays(value);
        return;
    end
    if isa(value, 'datetime')
        column = seconds(value - value(1));
        column = column(:);
        return;
    end
    if isa(value, 'duration')
        column = seconds(value);
        column = column(:);
        return;
    end
    if isnumeric(value) || islogical(value)
        column = double(value(:));
        return;
    end
    if iscell(value)
        tmp = cellfun(@ensure_numeric_column, value, 'UniformOutput', false);
        tmp = tmp(~cellfun(@isempty, tmp));
        if isempty(tmp)
            column = [];
            return;
        end
        try
            column = vertcat(tmp{:});
        catch
            try
                column = cell2mat(tmp(:));
            catch
                column = [];
            end
        end
        column = column(:);
        return;
    end
    if isstring(value) || ischar(value)
        column = [];
        return;
    end
    if isstruct(value)
        [~, column] = struct_to_series(value);
        return;
    end
end

function timeHours = normalize_time_vector(timeInput, fieldName)
    if nargin < 2
        fieldName = '';
    end
    timeHours = [];
    if isempty(timeInput)
        return;
    end
    if isa(timeInput, 'datetime')
        baseSeconds = seconds(timeInput - timeInput(1));
        timeHours = baseSeconds / 3600;
        return;
    end
    if isa(timeInput, 'duration')
        timeHours = seconds(timeInput) / 3600;
        return;
    end

    numericTime = ensure_numeric_column(timeInput);
    if isempty(numericTime)
        return;
    end

    numericTime = numericTime(:);
    numericTime = numericTime - numericTime(1);
    timeHours = infer_time_units(numericTime, fieldName);
end

function timeHours = infer_time_units(numericTime, fieldName)
    timeHours = [];
    if isempty(numericTime)
        return;
    end

    lowerName = lower(char(fieldName));
    if contains(lowerName, 'hour')
        timeHours = numericTime;
        return;
    elseif contains(lowerName, 'minute')
        timeHours = numericTime / 60;
        return;
    elseif contains(lowerName, 'second') || contains(lowerName, 'timestamp')
        timeHours = numericTime / 3600;
        return;
    elseif contains(lowerName, 'day')
        timeHours = numericTime * 24;
        return;
    end

    diffs = diff(numericTime);
    diffs = diffs(~isnan(diffs));
    if isempty(diffs)
        timeHours = numericTime / 3600;
        return;
    end
    dt = median(abs(diffs));

    if dt >= 3595
        timeHours = numericTime / 3600;
    elseif dt >= 59 && dt <= 61
        timeHours = numericTime / 60;
    elseif dt >= 0.0416 && dt <= 0.0420
        timeHours = numericTime * 24;
    elseif dt >= 0.9 && dt <= 1.1
        timeHours = numericTime;
    elseif dt > 0 && dt < 0.9
        timeHours = numericTime;
    else
        timeHours = numericTime / 3600;
    end
end
function [alignedTime, alignedData] = align_and_clip_series(timeVec, dataVec, horizonHours)
    alignedTime = [];
    alignedData = [];
    if isempty(timeVec) || isempty(dataVec)
        return;
    end

    mask = true(size(timeVec));
    if nargin >= 3 && ~isempty(horizonHours)
        mask = (timeVec - timeVec(1)) <= horizonHours;
        if nnz(mask) < 2
            mask = true(size(timeVec));
        end
    end

    alignedTime = timeVec(mask);
    alignedData = dataVec(mask);
end

function [strategy, meta] = derive_strategy(pvData, loadData, priceData)
    strategy = struct('timeHours', [], 'bestActionKW', [], 'socTrajectory', []);
    meta = struct('chargeHours', 0, 'dischargeHours', 0, 'idleHours', 0, ...
        'priceThreshold', NaN, 'capacity_kWh', 0);

    if isempty(pvData.power) || isempty(loadData.power) || isempty(priceData.price)
        return;
    end

    horizonHours = 72;
    % Unify inputs to hourly timetable and select first horizon window
    tt = preprocess_timeseries(pvData, loadData, priceData);
    if isempty(tt) || height(tt) == 0
        return;
    end
    t0 = tt.Properties.RowTimes(1);
    ttSel = tt(tt.Properties.RowTimes <= t0 + hours(horizonHours), :);
    if height(ttSel) < 2
        ttSel = tt; % fallback: use all available rows
    end
    timeHours   = hours(ttSel.Properties.RowTimes - t0);
    loadPower   = ttSel.Load_Power;
    pvPower     = ttSel.PV_Power;
    priceSeries = ttSel.Price_Buy;

    if isempty(timeHours) || isempty(loadPower) || isempty(pvPower) || isempty(priceSeries)
        return;
    end

    seriesLength = min([numel(timeHours), numel(loadPower), numel(pvPower), numel(priceSeries)]);
    timeHours = timeHours(1:seriesLength);
    loadPower = loadPower(1:seriesLength);
    pvPower = pvPower(1:seriesLength);
    priceSeries = priceSeries(1:seriesLength);

    netLoad = loadPower - pvPower;
    priceThreshold = median(priceSeries, 'omitnan');
    priceFlag = priceSeries >= priceThreshold;

    candidateActions = (-8000:2000:8000);
    soc = 50;
    capacity_kWh = 224;  % Approximated 280Ah @ 800V
    socTrajectory = zeros(numel(netLoad), 1);
    bestActions = zeros(numel(netLoad), 1);

    rewardFuncAvailable = exist('calculate_economic_reward', 'file') == 2;

    for idx = 1:numel(netLoad)
        bestReward = -Inf;
        chosenAction = 0;
        priceNorm = double(priceFlag(idx));
        for action = candidateActions
            if rewardFuncAvailable
                rewardValue = calculate_economic_reward(netLoad(idx), action, priceNorm, soc);
            else
                rewardValue = heuristic_reward(netLoad(idx), action, priceNorm);
            end
            if rewardValue > bestReward
                bestReward = rewardValue;
                chosenAction = action;
            end
        end
        bestActions(idx) = chosenAction / 1000; % convert to kW for plotting
        soc = soc + (chosenAction / 1000) / capacity_kWh * 100;
        soc = min(100, max(0, soc));
        socTrajectory(idx) = soc;
    end

    meta.chargeHours = nnz(bestActions > 0);
    meta.dischargeHours = nnz(bestActions < 0);
    meta.idleHours = nnz(abs(bestActions) < 0.1);
    meta.priceThreshold = priceThreshold;
    meta.capacity_kWh = capacity_kWh;

    strategy.timeHours = timeHours;
    strategy.bestActionKW = bestActions;
    strategy.socTrajectory = socTrajectory;
end

function value = heuristic_reward(netLoad, action, priceFlag)
    priceWeight = priceFlag * 2 + 1;
    loadComponent = -abs(netLoad + action);
    actionPenalty = -0.1 * abs(action);
    value = priceWeight * loadComponent + actionPenalty;
end

function summaryLines = compose_summary_text(rewards, trainingResults, strategyMeta, pvData, loadData, priceData)
    totalEpisodes = numel(rewards);
    initialAvg = safe_mean(rewards(1:min(10, totalEpisodes)));
    finalAvg = safe_mean(rewards(max(1, totalEpisodes - min(10, totalEpisodes) + 1):end));
    bestReward = max(rewards, [], 'omitnan');
    worstReward = min(rewards, [], 'omitnan');

    trainingTime = collect_field(trainingResults, 'training_time');
    trainingMinutes = trainingTime / 60;

    pvEnergy = estimate_energy(pvData);
    loadEnergy = estimate_energy(loadData);

    peakPrice = max(priceData.price, [], 'omitnan');
    valleyPrice = min(priceData.price, [], 'omitnan');

    summaryLines = {
        '════════════════════════════════════════════════════════'
        '                   SAC Training & Strategy Summary'
        '════════════════════════════════════════════════════════'
        ''
        sprintf('Episodes trained        : %d', totalEpisodes)
        sprintf('Initial 10-episode mean : %.2f', initialAvg)
        sprintf('Final 10-episode mean   : %.2f', finalAvg)
        sprintf('Best / worst reward     : %.2f / %.2f', bestReward, worstReward)
        sprintf('Total training time     : %.1f minutes', trainingMinutes)
        ''
        sprintf('Total PV energy         : %.1f MWh', pvEnergy / 1000)
        sprintf('Total load energy       : %.1f MWh', loadEnergy / 1000)
        sprintf('Price range             : %.2f - %.2f $/kWh', valleyPrice, peakPrice)
        sprintf('Strategy price threshold: %.2f $/kWh', strategyMeta.priceThreshold)
        ''
        sprintf('Recommended charge hrs  : %d hours', strategyMeta.chargeHours)
        sprintf('Recommended discharge hrs: %d hours', strategyMeta.dischargeHours)
        sprintf('Idle duration           : %d hours', strategyMeta.idleHours)
        sprintf('Estimated battery size  : %.0f kWh', strategyMeta.capacity_kWh)
        ''
        'Strategy tips:'
        '  • Low price + low net load  -> charge the battery'
        '  • High price + high net load -> discharge to support load'
        '  • Negative net load         -> consider charging or idling'
        ''
        sprintf('Figure generated at     : %s', char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')))
        '════════════════════════════════════════════════════════'
    };
end

function energyWh = estimate_energy(tsData)
    if isempty(tsData.power)
        energyWh = 0;
        return;
    end
    % Assuming hourly resolution
    energyWh = sum(tsData.power, 'omitnan');
end

function fig = create_power_flow_analysis(opts, pvData, loadData, priceData)
    fig = make_figure('Power Flow Analysis', opts.showFigures, [200, 200, 1400, 900]);
    tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    horizonHours = 72;
    [timeHours, loadPower] = align_and_clip_series(loadData.timeHours, loadData.power, horizonHours);
    [~, pvPower] = align_and_clip_series(pvData.timeHours, pvData.power, horizonHours);
    [~, ~] = align_and_clip_series(priceData.timeHours, priceData.price, horizonHours);

    if isempty(timeHours) || isempty(loadPower) || isempty(pvPower)
        timeHours = loadData.timeHours;
        loadPower = loadData.power;
        pvPower = pvData.power;
        if isempty(timeHours) || isempty(loadPower) || isempty(pvPower)
            subplot(2, 3, [1, 6]);
            text(0.5, 0.5, 'No power data available', 'HorizontalAlignment', 'center', 'FontSize', 12);
            axis off;
            return;
        end
    end

    loadPowerKW = loadPower / 1000;
    pvPowerKW = pvPower / 1000;
    netLoadKW = (loadPower - pvPower) / 1000;

    nexttile;
    area(timeHours, [pvPowerKW, loadPowerKW], 'FaceAlpha', 0.7);
    xlabel('Time (hours)');
    ylabel('Power (kW)');
    title('Generation vs Demand');
    legend({'PV Generation', 'Load Demand'}, 'Location', 'best');
    grid on;
    xlim([timeHours(1), min(timeHours(end), timeHours(1) + 72)]);

    nexttile;
    pv_sorted = sort(pvPowerKW, 'descend');
    load_sorted = sort(loadPowerKW, 'descend');
    duration = (1:length(pv_sorted)) / length(pv_sorted) * 100;

    plot(duration, pv_sorted, 'b-', 'LineWidth', 2, 'DisplayName', 'PV Generation');
    hold on;
    plot(duration, load_sorted, 'r-', 'LineWidth', 2, 'DisplayName', 'Load Demand');
    xlabel('Duration (%)');
    ylabel('Power (kW)');
    title('Power Duration Curves');
    legend('Location', 'best');
    grid on;

    nexttile;
    positive_power = max(0, netLoadKW);
    negative_power = min(0, netLoadKW);

    area(timeHours, positive_power, 'FaceColor', 'red', 'FaceAlpha', 0.7, 'DisplayName', 'Deficit');
    hold on;
    area(timeHours, negative_power, 'FaceColor', 'green', 'FaceAlpha', 0.7, 'DisplayName', 'Surplus');
    plot(timeHours, zeros(size(timeHours)), 'k--', 'LineWidth', 0.5);
    xlabel('Time (hours)');
    ylabel('Net Power (kW)');
    title('Energy Balance');
    legend('Location', 'best');
    grid on;
    xlim([timeHours(1), min(timeHours(end), timeHours(1) + 72)]);

    nexttile;
    hourly_avg_load = calculate_hourly_averages(loadPowerKW, timeHours);
    hours = 0:23;

    bar(hours, hourly_avg_load, 'FaceColor', [0.7, 0.3, 0.3], 'FaceAlpha', 0.8);
    xlabel('Hour of Day');
    ylabel('Average Load (kW)');
    title('Daily Load Profile');
    grid on;
    xlim([-0.5, 23.5]);

    nexttile;
    capacity_factor_pv = pvPowerKW / max(pvPowerKW);
    capacity_factor_load = loadPowerKW / max(loadPowerKW);
    capacity_factor_pv(isnan(capacity_factor_pv)) = 0;
    capacity_factor_load(isnan(capacity_factor_load)) = 0;

    plot(timeHours, capacity_factor_pv * 100, 'b-', 'LineWidth', 1.5, 'DisplayName', 'PV Capacity Factor');
    hold on;
    plot(timeHours, capacity_factor_load * 100, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Load Factor');
    xlabel('Time (hours)');
    ylabel('Capacity Factor (%)');
    title('Capacity Factor Analysis');
    legend('Location', 'best');
    grid on;
    xlim([timeHours(1), min(timeHours(end), timeHours(1) + 72)]);

    nexttile;
    create_power_statistics_panel(loadPowerKW, pvPowerKW, netLoadKW);

    sgtitle('Power Flow Analysis', 'FontSize', 14, 'FontWeight', 'bold');
end

function create_power_statistics_panel(pvPower, loadPower, netLoad)
    axis off;

    avg_pv = mean(pvPower);
    avg_load = mean(loadPower);
    max_pv = max(pvPower);
    max_load = max(loadPower);
    energy_balance = sum(netLoad);
    self_consumption = sum(min(pvPower, loadPower)) / sum(pvPower) * 100;

    text_str = sprintf(['Power Statistics:\n\n' ...
                       'Average PV power: %.1f kW\n' ...
                       'Average load power: %.1f kW\n' ...
                       'Peak PV power: %.1f kW\n' ...
                       'Peak load power: %.1f kW\n' ...
                       'Energy balance: %.1f kWh\n' ...
                       'Self-consumption: %.1f%%'], ...
                       avg_pv, avg_load, max_pv, max_load, energy_balance, self_consumption);

    text(0.05, 0.95, text_str, 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'FontSize', 9, ...
         'FontName', 'FixedWidth', 'BackgroundColor', [0.95, 0.95, 0.95]);
end

function hourly_avg = calculate_hourly_averages(power_data, time_plot)
    % Calculate average power for each hour of the day

    hourly_avg = zeros(24, 1);

    % Convert time to hours of day
    hours_of_day = mod(time_plot * 24, 24);

    for hour = 0:23
        hour_mask = (hours_of_day >= hour) & (hours_of_day < hour + 1);
        if any(hour_mask)
            hourly_avg(hour + 1) = mean(power_data(hour_mask));
        end
    end
end

function savedCount = save_figures(figures, opts)
    savedCount = 0;
    labels = {'training_overview', 'microgrid_profiles', 'summary_report', ...
              'power_flow_analysis', 'economic_analysis', 'system_performance', ...
              'battery_performance', 'battery_summary'};
    for idx = 1:numel(figures)
        fig = figures{idx};
        if ~ishandle(fig)
            continue;
        end
        savedCount = savedCount + 1;

        % 优先使用图对象的自定义标签，其次退回到预设labels，最后用通用命名
        labelName = '';
        if idx <= numel(labels)
            labelName = labels{idx};
        end
        try
            ud = get(fig, 'UserData');
            if isstruct(ud) && isfield(ud, 'label') && ~isempty(ud.label)
                labelName = char(ud.label);
            end
        catch
        end
        if isempty(labelName)
            labelName = sprintf('figure_%02d', idx);
        end

        if opts.saveFigures
            filename = sprintf('%s_%s_%s.png', opts.filePrefix, labelName, opts.timestamp);
            filepath = fullfile(opts.outputDir, filename);
            try
                viz_export(fig, filepath);
                fprintf('    - 已保存: %s\n', filepath);
            catch ME
                fprintf('    ⚠ 保存失败 (%s): %s\n', labelName, ME.message);
            end
        end
        if opts.closeAfterSave
            close(fig);
        end
    end
end

function fig = create_economic_analysis(opts, ~, loadData, priceData)
    fig = make_figure('Economic Analysis', opts.showFigures, [300, 300, 1400, 900]);
    tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Prepare economic data
    horizonHours = 168; % One week for economic analysis
    [timeHours, loadPower] = align_and_clip_series(loadData.timeHours, loadData.power, horizonHours);
    [priceTime, priceSeries] = align_and_clip_series(priceData.timeHours, priceData.price, horizonHours);

    if isempty(timeHours) || isempty(loadPower) || isempty(priceSeries)
        timeHours = loadData.timeHours;
        loadPower = loadData.power;
        priceTime = priceData.timeHours;
        priceSeries = priceData.price;
        if isempty(timeHours) || isempty(loadPower) || isempty(priceSeries)
                    nexttile(1);
            text(0.5, 0.5, 'No economic data available', 'HorizontalAlignment', 'center', 'FontSize', 12);
            axis off;
            return;
        end
    end

    seriesLength = min([numel(timeHours), numel(loadPower), numel(priceSeries), numel(priceTime)]);
    timeHours = timeHours(1:seriesLength);
    loadPower = loadPower(1:seriesLength);
    priceSeries = priceSeries(1:seriesLength);
    priceTime = priceTime(1:seriesLength);

    % Calculate economic metrics
    loadPowerKW = loadPower / 1000;
    hourly_cost = loadPowerKW .* priceSeries; % $/hour
    cumulative_cost = cumsum(hourly_cost);

    nexttile;
    plot(priceTime, priceSeries, 'g-', 'LineWidth', 1.5);
    xlabel('Time (hours)');
    ylabel('Price ($/kWh)');
    title('Electricity Price Profile');
    grid on;
    xlim([priceTime(1), min(priceTime(end), priceTime(1) + horizonHours)]);

    nexttile;
    plot(timeHours, hourly_cost, 'r-', 'LineWidth', 1.5);
    xlabel('Time (hours)');
    ylabel('Hourly Cost ($)');
    title('Hourly Electricity Cost');
    grid on;
    xlim([timeHours(1), min(timeHours(end), timeHours(1) + horizonHours)]);

    nexttile;
    plot(timeHours, cumulative_cost, 'b-', 'LineWidth', 2);
    xlabel('Time (hours)');
    ylabel('Cumulative Cost ($)');
    title('Accumulated Cost');
    grid on;
    xlim([timeHours(1), min(timeHours(end), timeHours(1) + horizonHours)]);

    nexttile;
    create_daily_cost_analysis(timeHours, hourly_cost);

    nexttile;
    scatter(priceSeries, loadPowerKW, 20, 'filled', 'MarkerFaceAlpha', 0.6);
    xlabel('Price ($/kWh)');
    ylabel('Load Power (kW)');
    title('Price vs Load Correlation');
    grid on;

    % Subplot 6: Economic statistics
    nexttile;
    create_economic_statistics_panel(priceSeries, hourly_cost, cumulative_cost);

    sgtitle('Economic Analysis', 'FontSize', 14, 'FontWeight', 'bold');
end

function create_daily_cost_analysis(timeHours, hourlyCost)
    % Create daily cost analysis bar chart

    days = floor(timeHours / 24) + 1;
    unique_days = unique(days);

    daily_costs = zeros(size(unique_days));

    for i = 1:length(unique_days)
        day_mask = (days == unique_days(i));
        daily_costs(i) = sum(hourlyCost(day_mask));
    end

    bar(unique_days, daily_costs, 'FaceColor', [0.3, 0.7, 0.3], 'FaceAlpha', 0.8);
    xlabel('Day');
    ylabel('Daily Cost ($)');
    title('Daily Cost Analysis');
    grid on;
end

function create_economic_statistics_panel(priceData, hourlyCost, cumulativeCost)
    % Create economic statistics text panel

    axis off;

    avg_price = mean(priceData);
    max_price = max(priceData);
    min_price = min(priceData);
    total_cost = cumulativeCost(end);
    avg_hourly_cost = mean(hourlyCost);
    daily_avg_cost = avg_hourly_cost * 24;

    text_str = sprintf(['Economic Statistics:\n\n' ...
                       'Average price: $%.3f/kWh\n' ...
                       'Price range: $%.3f-$%.3f/kWh\n' ...
                       'Total cost: $%.2f\n' ...
                       'Average hourly cost: $%.2f\n' ...
                       'Average daily cost: $%.2f'], ...
                       avg_price, min_price, max_price, total_cost, avg_hourly_cost, daily_avg_cost);

    text(0.05, 0.95, text_str, 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'FontSize', 9, ...
         'FontName', 'FixedWidth', 'BackgroundColor', [0.95, 0.95, 0.95]);
end

function fig = create_system_performance(opts, pvData, loadData, ~)
    fig = make_figure('System Performance Analysis', opts.showFigures, [400, 400, 1400, 900]);
    tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Prepare performance data
    horizonHours = 168; % One week for performance analysis
    [timeHours, loadPower] = align_and_clip_series(loadData.timeHours, loadData.power, horizonHours);
    [~, pvPower] = align_and_clip_series(pvData.timeHours, pvData.power, horizonHours);

    if isempty(timeHours) || isempty(loadPower) || isempty(pvPower)
        % Fallback to available data
        timeHours = loadData.timeHours;
        loadPower = loadData.power;
        pvPower = pvData.power;
        if isempty(timeHours) || isempty(loadPower) || isempty(pvPower)
            subplot(2, 3, [1, 6]);
            text(0.5, 0.5, 'No performance data available', 'HorizontalAlignment', 'center', 'FontSize', 12);
            axis off;
            return;
        end
    end

    % Convert to kW
    loadPowerKW = loadPower / 1000;
    pvPowerKW = pvPower / 1000;

    % Calculate performance metrics
    capacity_factor_pv = pvPowerKW / max(pvPowerKW);
    capacity_factor_load = loadPowerKW / max(loadPowerKW);
    capacity_factor_pv(isnan(capacity_factor_pv)) = 0;
    capacity_factor_load(isnan(capacity_factor_load)) = 0;

    energy_balance = cumsum(pvPowerKW - loadPowerKW);
    system_efficiency = min(pvPowerKW, loadPowerKW) ./ max(pvPowerKW, loadPowerKW);
    system_efficiency(isnan(system_efficiency)) = 0;

    % Subplot 1: Capacity factors
    nexttile;
    plot(timeHours, capacity_factor_pv * 100, 'b-', 'LineWidth', 1.5, 'DisplayName', 'PV Capacity Factor');
    hold on;
    plot(timeHours, capacity_factor_load * 100, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Load Factor');
    xlabel('Time (hours)');
    ylabel('Capacity Factor (%)');
    title('System Capacity Factors');
    legend('Location', 'best');
    grid on;
    xlim([timeHours(1), min(timeHours(end), timeHours(1) + horizonHours)]);

    % Subplot 2: Energy balance
    nexttile;
    plot(timeHours, energy_balance, 'k-', 'LineWidth', 2);
    xlabel('Time (hours)');
    ylabel('Cumulative Energy Balance (kWh)');
    title('Cumulative Energy Balance');
    grid on;
    xlim([timeHours(1), min(timeHours(end), timeHours(1) + horizonHours)]);

    % Subplot 3: System efficiency
    nexttile;
    plot(timeHours, system_efficiency * 100, 'm-', 'LineWidth', 1.5);
    xlabel('Time (hours)');
    ylabel('System Efficiency (%)');
    title('System Matching Efficiency');
    grid on;
    xlim([timeHours(1), min(timeHours(end), timeHours(1) + horizonHours)]);

    % Subplot 4: Performance distribution
    nexttile; % tile 4
    histogram(capacity_factor_pv * 100, 20, 'FaceColor', [0.3, 0.5, 0.8], 'FaceAlpha', 0.7);
    xlabel('PV Capacity Factor (%)');
    ylabel('Frequency');
    title('PV Capacity Factor Distribution');
    grid on;

    % Subplot 5: Efficiency distribution
    nexttile;
    histogram(system_efficiency * 100, 20, 'FaceColor', [0.8, 0.3, 0.8], 'FaceAlpha', 0.7);
    xlabel('System Efficiency (%)');
    ylabel('Frequency');
    title('System Efficiency Distribution');
    grid on;

    % Subplot 6: Performance statistics
    nexttile;
    create_performance_statistics_panel(pvPowerKW, loadPowerKW, capacity_factor_pv, capacity_factor_load);

    sgtitle('System Performance Analysis', 'FontSize', 14, 'FontWeight', 'bold');
end

function create_performance_statistics_panel(pvPower, loadPower, capacityFactorPV, capacityFactorLoad)
    % Create performance statistics text panel

    axis off;

    avg_pv = mean(pvPower);
    avg_load = mean(loadPower);
    avg_capacity_factor_pv = mean(capacityFactorPV) * 100;
    avg_capacity_factor_load = mean(capacityFactorLoad) * 100;
    energy_balance = sum(pvPower) - sum(loadPower);
    self_consumption = sum(min(pvPower, loadPower)) / sum(pvPower) * 100;

    text_str = sprintf(['Performance Statistics:\n\n' ...
                       'Average PV power: %.1f kW\n' ...
                       'Average load power: %.1f kW\n' ...
                       'PV capacity factor: %.1f%%\n' ...
                       'Load factor: %.1f%%\n' ...
                       'Energy balance: %.1f kWh\n' ...
                       'Self-consumption: %.1f%%'], ...
                       avg_pv, avg_load, avg_capacity_factor_pv, avg_capacity_factor_load, ...
                       energy_balance, self_consumption);

    text(0.05, 0.95, text_str, 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'FontSize', 9, ...
         'FontName', 'FixedWidth', 'BackgroundColor', [0.95, 0.95, 0.95]);
end

function fig = make_figure(name, showFigure, position)
    if nargin < 3
        position = [100, 100, 1200, 800];
    end
    visibility = ternary(showFigure, 'on', 'off');
    fig = figure('Name', name, 'Color', 'w', 'Visible', visibility, 'Position', position);
end

function value = get_option(options, field, defaultValue)
    if isstruct(options) && isfield(options, field)
        value = options.(field);
    else
        value = defaultValue;
    end
end

function viz_init_style()
% Initialize global figure/axes styles for publication
set(groot,'defaultAxesFontName','Times New Roman');
set(groot,'defaultAxesFontSize',10);
set(groot,'defaultLineLineWidth',1.8);
set(groot,'defaultAxesLineWidth',1.0);
set(groot,'defaultLegendBox','off');
set(groot,'defaultFigureColor','w');
end

function viz_export(fig, filepath)
% Unified PNG export at 300 dpi (journal-ready bitmap)
[p,n,ext] = fileparts(filepath);
if isempty(ext) || ~strcmpi(ext,'.png')
    filepath = fullfile(p, [n '.png']);
end
% [0m[0m[0m[0m
%  [0m[0m[0m[0m[0m
%  [0m[0m[0m[0m[0m
%  [0m[0m[0m[0m[0m
%  [0m[0m[0m[0m[0m
%  [0m[0m[0m[0m[0m
% [0m[0m[0m[0m
% [0m[0m[0m[0m
% [0m[0m[0m[0m
% [0m[0m[0m[0m
% [0m[0m[0m[0m
% [0m[0m[0m[0m
% [0m[0m[0m[0m
% [0m[0m[0m[0m
if ~exist(p, 'dir')
    mkdir(p);
end
try
    %  [0m[0m[0m[0m[0m
    if exist('exportgraphics','file') == 2
        exportgraphics(fig, filepath, 'Resolution', 300, 'BackgroundColor', 'white');
    else
        set(fig, 'PaperPositionMode', 'auto');
        print(fig, filepath, '-dpng', '-r300');
    end
catch ME
    error('viz_export:saveFailed', '%s', ME.message);
end
end


function tt = preprocess_timeseries(pvProfile, loadProfile, priceProfile)
%PREPROCESS_TIMESERIES Unify PV/Load/Price into hourly timetable.
%   tt = preprocess_timeseries(pvProfile, loadProfile, priceProfile)
%   Inputs can be timeseries/timetable/table/struct/array-like (with time).
%   Output timetable has variables: PV_Power, Load_Power, Price_Buy and
%   row times at 1-hour resolution. Missing values are interpolated.

    % Use existing helpers to obtain time in hours and numeric columns
    [tp, pv]    = timeseries_to_array(pvProfile);
    [tl, loadp] = timeseries_to_array(loadProfile);
    [tpr, prc]  = timeseries_to_array(priceProfile);

    % Create datetimes from hours offset
    baseT = datetime(2000,1,1,0,0,0);
    if isempty(tp) && ~isempty(pv); tp = (0:numel(pv)-1)'; end
    if isempty(tl) && ~isempty(loadp); tl = (0:numel(loadp)-1)'; end
    if isempty(tpr) && ~isempty(prc); tpr = (0:numel(prc)-1)'; end

    ttpv   = timetable(baseT + hours(tp),  double(pv(:)),   'VariableNames', {'PV_Power'});
    ttload = timetable(baseT + hours(tl),  double(loadp(:)),'VariableNames', {'Load_Power'});
    ttpri  = timetable(baseT + hours(tpr), double(prc(:)),  'VariableNames', {'Price_Buy'});

    % Synchronize then retime to hourly mean
    try
        tt = synchronize(ttpv, ttload, ttpri, 'union', 'mean');
    catch
        tt = synchronize(ttpv, ttload, ttpri);
    end
    tt = retime(tt, 'hourly', 'mean');

    % Fill missing values robustly
    vars = tt.Properties.VariableNames;
    for k = 1:numel(vars)
        v = vars{k};
        tt.(v) = fillmissing(tt.(v), 'linear', 'EndValues', 'nearest');
        if any(isnan(tt.(v)))
            tt.(v) = fillmissing(tt.(v), 'previous');
            tt.(v) = fillmissing(tt.(v), 'next');
        end
        tt.(v)(isnan(tt.(v))) = 0; % final fallback
    end
end

function [p_batt_norm, p_grid_norm] = normalize_signs(p_batt, p_grid)
%NORMALIZE_SIGNS Enforce sign convention: batt charge>0, grid import>0.
    p_batt_norm = double(p_batt(:));
    p_grid_norm = double(p_grid(:));
    % Project prohibits selling to grid currently
    p_grid_norm = max(p_grid_norm, 0);
end

function p_grid = compute_grid_power(pv, loadp, p_batt)
%COMPUTE_GRID_POWER Grid import = load - PV - p_batt (clip to >=0).
    pv     = double(pv(:));
    loadp  = double(loadp(:));
    p_batt = double(p_batt(:));
    n = min([numel(pv), numel(loadp), numel(p_batt)]);
    pv = pv(1:n); loadp = loadp(1:n); p_batt = p_batt(1:n);
    p_grid = loadp - pv - p_batt;
    p_grid = max(p_grid, 0);
end

function viz_tou_shading(ax, touDef)
%VIZ_TOU_SHADING Shade TOU periods on given axes using semi-transparent patches.
% touDef rows: [start_hour, end_hour, R, G, B]
    if nargin < 1 || isempty(ax) || ~ishandle(ax); ax = gca; end
    if nargin < 2 || isempty(touDef)
        touDef = [
            1,  6,  0.92, 0.98, 1.00;  % valley (light blue)
            7, 10,  0.98, 0.92, 0.92;  % peak (light red)
           18, 22,  0.98, 0.92, 0.92;  % peak (light red)
        ];
    end
    axes(ax); hold(ax, 'on');
    yl = ylim(ax);
    for i = 1:size(touDef,1)
        x1 = touDef(i,1); x2 = touDef(i,2);
        c  = touDef(i,3:5);
        patch(ax, [x1 x2 x2 x1], [yl(1) yl(1) yl(2) yl(2)], c, ...
            'EdgeColor','none','FaceAlpha',0.15);
    end
    % push patches to the bottom
    try
        uistack(findobj(ax,'Type','patch'), 'bottom');
    catch
    end
end
function value = collect_field(data, field)
    value = [];
    if isempty(data)
        return;
    end
    try
        value = data.(field);
    catch
        value = [];
    end
    if isnumeric(value)
        value = double(value(:));
    end
end

function result = fetch_workspace_variable(workspace, candidates)
    result = [];
    if nargin < 2 || isempty(candidates)
        return;
    end
    candidates = string(candidates);
    for idx = 1:numel(candidates)
        name = strtrim(candidates(idx));
        if name == ""
            continue;
        end
        switch workspace
            case 'base'
                if evalin('base', sprintf('exist(''%s'', ''var'')', name))
                    result = evalin('base', name);
                    return;
                end
            case 'caller'
                if evalin('caller', sprintf('exist(''%s'', ''var'')', name))
                    result = evalin('caller', name);
                    return;
                end
            otherwise
                if isstruct(workspace) && isfield(workspace, name)
                    result = workspace.(name);
                    return;
                end
        end
    end
end

function out = safe_mean(vec)
    if isempty(vec)
        out = NaN;
    else
        out = mean(vec, 'omitnan');
    end
end

function out = ternary(condition, trueValue, falseValue)
    if condition
        out = trueValue;
    else
        out = falseValue;
    end
end

%% ========================================================================
%% BATTERY PERFORMANCE VISUALIZATION
%% ========================================================================

function fig = create_battery_performance(opts, batterySOC, batterySOH, batteryPower)
    %CREATE_BATTERY_PERFORMANCE Generate battery SOC/SOH performance visualization
    %   Creates a comprehensive 6-subplot figure showing battery state of charge,
    %   state of health, charging/discharging power, and statistical analysis
    %   for 30-day simulation data.
    %
    %   Inputs:
    %       opts - visualization options structure
    %       batterySOC - Battery state of charge time series or array
    %       batterySOH - Battery state of health time series or array
    %       batteryPower - Battery power time series or array (optional)

    fig = make_figure('电池性能分析', opts.showFigures, [400, 400, 1400, 900]);
    tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Extract and process time series data
    [socTime, socData] = extract_battery_series(batterySOC, '时间 (天)');
    [sohTime, sohData] = extract_battery_series(batterySOH, '时间 (天)');
    [powerTime, powerData] = extract_battery_series(batteryPower, '时间 (天)');

    % Convert time from hours to days if necessary
    socTime = ensure_time_in_days(socTime);
    sohTime = ensure_time_in_days(sohTime);
    powerTime = ensure_time_in_days(powerTime);

    % Filter to 30-day horizon
    horizonDays = 30;
    [socTime, socData] = filter_time_horizon(socTime, socData, horizonDays);
    [sohTime, sohData] = filter_time_horizon(sohTime, sohData, horizonDays);
    [powerTime, powerData] = filter_time_horizon(powerTime, powerData, horizonDays);

    % Smart detection: convert SOC/SOH to percentage if in 0-1 range
    socData = normalize_to_percentage(socData);
    sohData = normalize_to_percentage(sohData);

    % Convert power to kW if necessary
    if ~isempty(powerData)
        % Assume data > 1000 is in Watts, otherwise in kW
        if max(abs(powerData)) > 1000
            powerData = powerData / 1000;
        end
    end

    % Subplot 1: SOC 30-day time series
    nexttile;
    if isempty(socData)
        text(0.5, 0.5, '无SOC数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
        axis off;
    else
        plot(socTime, socData, 'b-', 'LineWidth', 2);
        hold on;
        yline(90, 'r--', 'LineWidth', 1, 'Alpha', 0.5, 'DisplayName', 'Max SOC');
        yline(10, 'r--', 'LineWidth', 1, 'Alpha', 0.5, 'DisplayName', 'Min SOC');
        hold off;
        xlabel('时间 (天)');
        ylabel('SOC (%)');
        title('电池SOC - 30天时序');
        legend('SOC', 'Location', 'best');
        grid on;
        xlim([0, min(max(socTime), horizonDays)]);
        ylim([0, 100]);
    end

    % Subplot 2: SOH 30-day time series
    nexttile;
    if isempty(sohData)
        text(0.5, 0.5, '无SOH数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
        axis off;
    else
        plot(sohTime, sohData, 'g-', 'LineWidth', 2);
        xlabel('时间 (天)');
        ylabel('SOH (%)');
        title('电池SOH - 30天时序');
        grid on;
        xlim([0, min(max(sohTime), horizonDays)]);
        ylim([max(80, min(sohData)-5), 100]);
    end

    % Subplot 3: Battery charging/discharging power
    nexttile;
    if isempty(powerData)
        % If no power data available, simulate or show placeholder
        text(0.5, 0.5, '无电池功率数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
        axis off;
    else
        positive_power = max(0, powerData);
        negative_power = min(0, powerData);

        area(powerTime, positive_power, 'FaceColor', 'red', 'FaceAlpha', 0.7, 'DisplayName', '放电');
        hold on;
        area(powerTime, negative_power, 'FaceColor', 'blue', 'FaceAlpha', 0.7, 'DisplayName', '充电');
        yline(0, 'k--', 'LineWidth', 0.5);
        hold off;
        xlabel('时间 (天)');
        ylabel('电池功率 (kW)');
        title('电池充放电功率');
        legend('Location', 'best');
        grid on;
        xlim([0, min(max(powerTime), horizonDays)]);
    end

    % Subplot 4: SOC distribution histogram
    nexttile;
    if isempty(socData)
        text(0.5, 0.5, '无SOC分布数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
        axis off;
    else
        histogram(socData, 20, 'FaceColor', [0.3, 0.5, 0.7], 'FaceAlpha', 0.8, 'EdgeColor', 'none');
        xlabel('SOC (%)');
        ylabel('频次');
        title('SOC分布直方图');
        grid on;
        xlim([0, 100]);
    end

    % Subplot 5: Battery efficiency analysis
    nexttile;
    if isempty(socData) || isempty(powerData)
        text(0.5, 0.5, '无效率分析数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
        axis off;
    else
        % Calculate round-trip efficiency based on SOC and power
        efficiency = calculate_battery_efficiency(powerData, socData);
        plot(powerTime, efficiency * 100, 'm-', 'LineWidth', 1.5);
        xlabel('时间 (天)');
        ylabel('效率 (%)');
        title('电池往返效率');
        grid on;
        xlim([0, min(max(powerTime), horizonDays)]);
        ylim([80, 100]);
    end

    % Subplot 6: Battery statistics panel
    nexttile;
    create_battery_statistics_panel_v2(socData, sohData, powerData);

    sgtitle('电池性能分析 - 30天仿真', 'FontSize', 14, 'FontWeight', 'bold');
end

function fig = create_battery_summary(opts, batterySOC, batterySOH, batterySOHDiff)
    %CREATE_BATTERY_SUMMARY Produce compact SOC/SOH/SOH diff visualisation with 3 subplots
    fig = make_figure('电池状态概要', opts.showFigures, [420, 420, 1200, 720]);
    tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    horizonDays = 30;

    [socTime, socData] = extract_battery_series(batterySOC, '时间 (天)');
    [sohTime, sohData] = extract_battery_series(batterySOH, '时间 (天)');
    [diffTime, diffData] = extract_battery_series(batterySOHDiff, '时间 (天)');

    socTime = ensure_time_in_days(socTime);
    sohTime = ensure_time_in_days(sohTime);
    diffTime = ensure_time_in_days(diffTime);

    [socTime, socData] = filter_time_horizon(socTime, normalize_to_percentage(socData), horizonDays);
    [sohTime, sohData] = filter_time_horizon(sohTime, normalize_to_percentage(sohData), horizonDays);
    [diffTime, diffData] = filter_time_horizon(diffTime, diffData, horizonDays);

    socTime = ensure_monotonic_time(socTime, numel(socData));
    sohTime = ensure_monotonic_time(sohTime, numel(sohData));
    diffTime = ensure_monotonic_time(diffTime, numel(diffData));

    % Subplot 1: SOC trend
    nexttile;
    if isempty(socData)
        axis_off_with_message('SOC 数据缺失');
    else
        plot(socTime, socData, 'LineWidth', 1.8, 'Color', [0.16, 0.44, 0.84]);
        hold on;
        yline([10, 90], 'r--', 'LineWidth', 1, 'Alpha', 0.4);
        hold off;
        xlabel('时间 (天)');
        ylabel('SOC (%)');
        title('SOC 趋势');
        grid on;
        ylim([max(0, min(socData, [], 'omitnan')-5), min(100, max(socData, [], 'omitnan')+5)]);
    end

    % Subplot 2: SOH trend
    nexttile;
    if isempty(sohData)
        axis_off_with_message('SOH 数据缺失');
    else
        plot(sohTime, sohData, 'LineWidth', 1.8, 'Color', [0.13, 0.64, 0.33]);
        xlabel('时间 (天)');
        ylabel('SOH (%)');
        title('SOH 趋势');
        grid on;
        ylim([max(70, min(sohData, [], 'omitnan')-2), 100]);
    end

    % Subplot 3: SOH differential
    nexttile;
    if isempty(diffData)
        axis_off_with_message('SOH变化 数据缺失');
    else
        stem(diffTime, diffData, 'filled', 'MarkerSize', 3, 'Color', [0.85, 0.33, 0.1]);
        xlabel('时间 (天)');
        ylabel('ΔSOH (%)');
        title('SOH 差分');
        grid on;
    end

    sgtitle('电池SOC/SOH状态概要', 'FontSize', 14, 'FontWeight', 'bold');
end

function [timeVec, dataVec] = extract_battery_series(input, ~)
    %EXTRACT_BATTERY_SERIES Extract time and data vectors from various input formats
    %   Handles timeseries, timetable, struct, and plain array inputs

    timeVec = [];
    dataVec = [];

    if isempty(input)
        return;
    end

    % Handle different input types
    if isa(input, 'timeseries')
        timeVec = input.Time(:);
        dataVec = double(input.Data(:));
    elseif istimetable(input)
        timeVec = seconds(input.Properties.RowTimes - input.Properties.RowTimes(1));
        varNames = input.Properties.VariableNames;
        if ~isempty(varNames)
            dataVec = double(input{:, 1});
        end
    elseif istable(input)
        varNames = input.Properties.VariableNames;
        % Try to find time and data columns
        timeIdx = find(contains(lower(varNames), {'time', 'hour', 'day'}), 1);
        dataIdx = find(~contains(lower(varNames), {'time', 'hour', 'day'}), 1);

        if ~isempty(timeIdx)
            timeVec = double(input{:, timeIdx});
        end
        if ~isempty(dataIdx)
            dataVec = double(input{:, dataIdx});
        end
    elseif isstruct(input)
        % Try common field names
        if isfield(input, 'Time')
            timeVec = double(input.Time(:));
        elseif isfield(input, 'time')
            timeVec = double(input.time(:));
        end

        if isfield(input, 'Data')
            dataVec = double(input.Data(:));
        elseif isfield(input, 'data')
            dataVec = double(input.data(:));
        elseif isfield(input, 'values')
            dataVec = double(input.values(:));
        end
    elseif isnumeric(input)
        dataVec = double(input(:));
    end

    % If no time vector, create sequential indices
    if isempty(timeVec) && ~isempty(dataVec)
        timeVec = (0:numel(dataVec)-1)';
    end

    % Ensure matching lengths
    if ~isempty(timeVec) && ~isempty(dataVec)
        minLen = min(numel(timeVec), numel(dataVec));
        timeVec = timeVec(1:minLen);
        dataVec = dataVec(1:minLen);
    end
end

function timeInDays = ensure_time_in_days(timeVec)
    %ENSURE_TIME_IN_DAYS Convert time vector to days if in seconds or hours

    if isempty(timeVec)
        timeInDays = [];
        return;
    end

    timeVec = double(timeVec(:));
    timeRange = max(timeVec) - min(timeVec);

    % Heuristic: if time range > 3600, assume seconds; if > 48, assume hours
    if timeRange > 86400  % More than 1 day in seconds
        timeInDays = timeVec / 86400;  % Convert seconds to days
    elseif timeRange > 48  % More than 2 days in hours
        timeInDays = timeVec / 24;  % Convert hours to days
    else
        timeInDays = timeVec;  % Already in days
    end

    % Normalize to start from 0
    timeInDays = timeInDays - timeInDays(1);
end

function [filteredTime, filteredData] = filter_time_horizon(timeVec, dataVec, horizonDays)
    %FILTER_TIME_HORIZON Filter data to specified time horizon

    if isempty(timeVec) || isempty(dataVec) || isempty(horizonDays)
        filteredTime = timeVec;
        filteredData = dataVec;
        return;
    end

    mask = timeVec <= horizonDays;
    filteredTime = timeVec(mask);
    filteredData = dataVec(mask);
end

function normalizedData = normalize_to_percentage(dataVec)
    %NORMALIZE_TO_PERCENTAGE Convert 0-1 range to 0-100% if necessary

    if isempty(dataVec)
        normalizedData = [];
        return;
    end

    % Smart detection: if max value <= 1, assume it's in 0-1 range
    if max(dataVec) <= 1.0
        normalizedData = dataVec * 100;
    else
        normalizedData = dataVec;
    end
end

function timeVec = ensure_monotonic_time(timeVec, desiredLength)
    if nargin < 2
        desiredLength = [];
    end

    if isempty(timeVec)
        if ~isempty(desiredLength) && desiredLength > 0
            timeVec = (0:desiredLength-1)';
        end
        return;
    end

    timeVec = double(timeVec(:));

    if ~isempty(desiredLength)
        desiredLength = max(1, desiredLength);
        if numel(timeVec) > desiredLength
            timeVec = timeVec(1:desiredLength);
        elseif numel(timeVec) < desiredLength
            extra = (numel(timeVec):desiredLength-1)';
            if isempty(extra)
                extra = [];
            end
            if isempty(timeVec)
                timeVec = (0:desiredLength-1)';
            else
                dt = infer_positive_step(diff(timeVec));
                timeVec = [timeVec; timeVec(end) + dt * (1:numel(extra))'];
            end
        end
    end

    if any(diff(timeVec) <= 0)
        dt = infer_positive_step(diff(timeVec));
        timeVec = timeVec(1) + (0:numel(timeVec)-1)' * dt;
    end
end

function dt = infer_positive_step(diffVec)
    diffVec = diffVec(:);
    diffVec = diffVec(diffVec > 0);
    if isempty(diffVec)
        dt = 1;
    else
        dt = median(diffVec);
        if dt <= 0
            dt = max(diffVec);
        end
        if dt <= 0
            dt = 1;
        end
    end
end

function efficiency = calculate_battery_efficiency(batteryPower, socData)
    %CALCULATE_BATTERY_EFFICIENCY Estimate battery round-trip efficiency
    %   Simple efficiency model based on power and SOC state

    if isempty(batteryPower) || isempty(socData)
        efficiency = [];
        return;
    end

    % Align lengths
    minLen = min(numel(batteryPower), numel(socData));
    batteryPower = batteryPower(1:minLen);
    socData = socData(1:minLen);

    % Base efficiency: 95%
    base_efficiency = 0.95;

    % Efficiency decreases with higher power and extreme SOC
    power_factor = 1 - 0.001 * abs(batteryPower);
    power_factor = max(0.9, min(1.0, power_factor));

    soc_factor = 1 - 0.1 * abs(socData/100 - 0.5);
    soc_factor = max(0.9, min(1.0, soc_factor));

    efficiency = base_efficiency * power_factor .* soc_factor;
    efficiency = max(0.80, min(0.98, efficiency));
end

function create_battery_statistics_panel_v2(socData, sohData, batteryPower)
    %CREATE_BATTERY_STATISTICS_PANEL_V2 Create battery statistics text panel

    axis off;


    if isempty(socData) && isempty(sohData)
        text(0.5, 0.5, '无电池统计数据', 'HorizontalAlignment', 'center', ...
             'FontSize', 12, 'Units', 'normalized');
        return;
    end

    % Calculate statistics
    if ~isempty(socData)
        avg_soc = mean(socData, 'omitnan');
        min_soc = min(socData);
        max_soc = max(socData);
        std_soc = std(socData, 'omitnan');
    else
        avg_soc = NaN; min_soc = NaN; max_soc = NaN; std_soc = NaN;
    end

    if ~isempty(sohData)
        avg_soh = mean(sohData, 'omitnan');
        min_soh = min(sohData);
        degradation = 100 - min_soh;
    else
        avg_soh = NaN; min_soh = NaN; degradation = NaN;
    end

    if ~isempty(batteryPower)
        max_charge = max(abs(batteryPower(batteryPower < 0)));
        max_discharge = max(batteryPower(batteryPower > 0));
        if isempty(max_charge), max_charge = 0; end
        if isempty(max_discharge), max_discharge = 0; end
    else
        max_charge = NaN; max_discharge = NaN;
    end

    % Build text string
    text_str = sprintf(['电池统计数据:\n\n' ...
                       'SOC 平均值: %.1f%%\n' ...
                       'SOC 范围: %.1f%% - %.1f%%\n' ...
                       'SOC 标准差: %.1f%%\n\n' ...
                       'SOH 平均值: %.1f%%\n' ...
                       'SOH 最小值: %.1f%%\n' ...
                       '累积衰减: %.2f%%\n\n' ...
                       '最大充电功率: %.1f kW\n' ...
                       '最大放电功率: %.1f kW'], ...
                       avg_soc, min_soc, max_soc, std_soc, ...
                       avg_soh, min_soh, degradation, ...
                       max_charge, max_discharge);

    text(0.05, 0.95, text_str, 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'FontSize', 9, ...
         'FontName', 'FixedWidth', 'BackgroundColor', [0.95, 0.95, 0.95]);
end

function axis_off_with_message(message)
    axis off;
    text(0.5, 0.5, message, 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', 'FontSize', 12, 'Units', 'normalized');
end


function fig = plot_fig_A_overview(opts, tt, touDef)
    %PLOT_FIG_A_OVERVIEW PV/Load/Price (Daily overview)
    fig = make_figure('Fig A: PV/Load vs TOU Price (Daily)', opts.showFigures, [100, 100, 1200, 500]);
    ax = axes(fig); hold(ax, 'on');
    n = min(24, height(tt));
    if n < 2
        axis(ax, 'off'); text(0.5,0.5,'数据不足','Units','normalized','HorizontalAlignment','center'); return; %#ok<UNRCH>
    end
    t = (1:n)';
    viz_tou_shading(ax, touDef);
    yyaxis(ax, 'left');
    plot(ax, t, tt.Load_Power(1:n), 'k-', 'LineWidth', 1.8, 'DisplayName', 'Load');
    plot(ax, t, tt.PV_Power(1:n),   '-', 'Color', [0.2, 0.6, 0.8], 'LineWidth', 1.8, 'DisplayName', 'PV');
    ylabel(ax, 'Power (kW)');
    yyaxis(ax, 'right');
    plot(ax, t, tt.Price_Buy(1:n), '--', 'Color', [0.85, 0.33, 0.1], 'LineWidth', 1.5, 'DisplayName', 'Price (buy)');
    ylabel(ax, 'Price ($/kWh)');
    xlabel(ax, 'Hour');
    legend(ax, {'Load','PV','Price (buy)'}, 'Location', 'northwest');
    title(ax, 'PV/Load vs TOU Price (Daily)'); grid(ax, 'on');
end

function fig = plot_fig_B_battery(opts, tt)
    %PLOT_FIG_B_BATTERY Battery power bars (charge/discharge) + SOC
    fig = make_figure('Fig B: Battery Power and SOC', opts.showFigures, [100, 100, 1200, 600]);
    tiledlayout(fig, 2, 1, 'TileSpacing','compact', 'Padding','compact');
    n = min(24, height(tt)); t = (1:n)';
    % 上: 充/放电双向柱
    ax1 = nexttile; hold(ax1, 'on');
    p = tt.Battery_Power(1:n);
    bar(ax1, t, max(p,0), 0.9, 'FaceColor', [0.3,0.6,0.85], 'EdgeColor','none', 'DisplayName','Charge');
    bar(ax1, t, min(p,0), 0.9, 'FaceColor', [0.9,0.4,0.4],  'EdgeColor','none', 'DisplayName','Discharge');
    yline(ax1, 0, 'k-'); ylabel(ax1, 'Power (kW)'); legend(ax1, 'Location','northwest'); grid(ax1,'on');
    title(ax1, 'Battery Charge/Discharge');
    % 下: SOC
    ax2 = nexttile; hold(ax2, 'on');
    if any(strcmp('Battery_SOC', tt.Properties.VariableNames)) && any(~isnan(tt.Battery_SOC))
        soc = tt.Battery_SOC(1:n);
        plot(ax2, t, soc, 'LineWidth', 2, 'Color', [0.2,0.2,0.2]);
        yline(ax2, [0.2 0.9], '--', 'Color', [0.5,0.5,0.5]);
        ylim(ax2, [0 1]); ylabel(ax2, 'SOC (pu)');
    else
        axis(ax2, 'off'); text(0.5,0.5,'缺少 Battery_SOC','Units','normalized','HorizontalAlignment','center');
    end
    xlabel(ax2, 'Hour'); grid(ax2, 'on');
end

function fig = plot_fig_C_grid_exchange(opts, tt)
    %PLOT_FIG_C_GRID_EXCHANGE Grid import (area), no export per spec
    fig = make_figure('Fig C: Grid Import (Area)', opts.showFigures, [100, 100, 1200, 400]);
    ax = axes(fig); hold(ax, 'on');
    n = min(24, height(tt)); t = (1:n)';
    if any(strcmp('Grid_Power', tt.Properties.VariableNames))
        p_grid = max(tt.Grid_Power(1:n), 0);
    else
        p_grid = compute_grid_power(tt.PV_Power(1:n), tt.Load_Power(1:n), tt.Battery_Power(1:n));
    end
    area(ax, t, p_grid, 'FaceAlpha', 0.6, 'FaceColor', [0.3, 0.5, 0.8], 'EdgeColor','none');
    yline(ax, 0, 'k-'); ylabel(ax, 'Power (kW)'); xlabel(ax, 'Hour');
    legend(ax, {'Grid Import (+)'}, 'Location', 'northwest'); grid(ax, 'on');
end

function fig = plot_fig_D_supply_stack(opts, tt)
    %PLOT_FIG_D_SUPPLY_STACK Supply stack: PV + Batt Discharge + Grid Import -> Load
    fig = make_figure('Fig D: Supply Stack', opts.showFigures, [100, 100, 1200, 500]);
    ax = axes(fig); hold(ax, 'on');
    n = min(24, height(tt)); t = (1:n)';
    pv_supply = tt.PV_Power(1:n);
    batt_dis = max(-tt.Battery_Power(1:n), 0);
    if any(strcmp('Grid_Power', tt.Properties.VariableNames))
        grid_imp = max(tt.Grid_Power(1:n), 0);
    else
        grid_imp = compute_grid_power(tt.PV_Power(1:n), tt.Load_Power(1:n), tt.Battery_Power(1:n));
    end
    area(ax, t, [pv_supply, batt_dis, grid_imp], 'LineStyle','none');
    colormap(ax, [0.2 0.6 0.8; 0.9 0.4 0.4; 0.3 0.5 0.8]);
    alpha(0.8);
    plot(ax, t, tt.Load_Power(1:n), 'k-', 'LineWidth', 2, 'DisplayName','Load');
    ylabel(ax, 'Power (kW)'); xlabel(ax, 'Hour');
    legend(ax, {'PV','Battery Discharge','Grid Import','Load'}, 'Location', 'northwest');
    grid(ax, 'on');
end
