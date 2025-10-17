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
    opts.timestamp      = char(get_option(options, 'timestamp', datestr(now, 'yyyymmdd_HHMMss')));

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
    figures{end+1} = create_microgrid_profiles(opts, pvData, loadData, priceData, strategy);
    figures{end+1} = create_strategy_summary(opts, rewardSeries, trainingResults, strategyMeta, ...
        pvData, loadData, priceData);

    savedCount = save_figures(figures, opts);
    fprintf('  ✓ 图像生成完成 (共%d个)\n\n', savedCount);
end

function fig = create_training_overview(opts, rewards, episodes, avgReward, episodeSteps, q0Series, trainingResults)
    fig = make_figure('SAC训练总览', opts.showFigures, [100, 100, 1400, 800]);
    tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    subplot(2,2,1);
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

    subplot(2,2,1);
    if isempty(episodeSteps)
        text(0.5, 0.5, '无Episode步数数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
    else
        bar(episodes, episodeSteps, 'FaceColor', [0.2, 0.6, 0.6], 'EdgeColor', 'none');
        xlabel('Episode');
        ylabel('Steps');
        title('每回合仿真步数');
        grid on;
    end

    subplot(2,2,1);
    if isempty(rewards)
        text(0.5, 0.5, '无奖励分布数据', 'HorizontalAlignment', 'center', 'FontSize', 12);
    else
        histogram(rewards, 'FaceColor', [0.4, 0.4, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
        xlabel('Reward');
        ylabel('Frequency');
        title('奖励分布');
        grid on;
    end

    subplot(2,2,1);
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

function fig = create_microgrid_profiles(opts, pvData, loadData, priceData, strategy)
    fig = make_figure('微电网运行概览', opts.showFigures, [150, 150, 1400, 900]);
    tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    subplot(2,2,1);
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

    subplot(2,2,1);
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

    subplot(2,2,1);
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
    [timeHours, loadPower] = align_and_clip_series(loadData.timeHours, loadData.power, horizonHours);
    [~, pvPower] = align_and_clip_series(pvData.timeHours, pvData.power, horizonHours);
    [~, priceSeries] = align_and_clip_series(priceData.timeHours, priceData.price, horizonHours);

    if isempty(timeHours) || isempty(loadPower) || isempty(pvPower) || isempty(priceSeries)
        return;
    end

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
        '                     SAC 训练与策略总结'
        '════════════════════════════════════════════════════════'
        ''
        sprintf('训练回合数     : %d', totalEpisodes)
        sprintf('初始10回合均值 : %.2f', initialAvg)
        sprintf('末尾10回合均值 : %.2f', finalAvg)
        sprintf('最佳/最差奖励  : %.2f / %.2f', bestReward, worstReward)
        sprintf('总训练时长     : %.1f 分钟', trainingMinutes)
        ''
        sprintf('光伏总能量     : %.1f MWh', pvEnergy / 1000)
        sprintf('负载总能量     : %.1f MWh', loadEnergy / 1000)
        sprintf('电价区间       : %.2f - %.2f $/kWh', valleyPrice, peakPrice)
        sprintf('策略电价阈值   : %.2f $/kWh', strategyMeta.priceThreshold)
        ''
        sprintf('建议充电时长   : %d 小时', strategyMeta.chargeHours)
        sprintf('建议放电时长   : %d 小时', strategyMeta.dischargeHours)
        sprintf('保持待机时长   : %d 小时', strategyMeta.idleHours)
        sprintf('电池容量估计   : %.0f kWh', strategyMeta.capacity_kWh)
        ''
        '策略提示:'
        '  • 低价+净负荷低 -> 充电蓄能'
        '  • 高价+净负荷高 -> 放电削峰'
        '  • 净负荷为负    -> 适度充电或待机'
        ''
        sprintf('图像生成时间   : %s', datestr(now, 'yyyy-mm-dd HH:MM:SS'))
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

function savedCount = save_figures(figures, opts)
    savedCount = 0;
    labels = {'training_overview', 'microgrid_profiles', 'summary_report'};
    for idx = 1:numel(figures)
        fig = figures{idx};
        if ~ishandle(fig)
            continue;
        end
        savedCount = savedCount + 1;
        if opts.saveFigures
            filename = sprintf('%s_%s_%s.%s', opts.filePrefix, labels{idx}, opts.timestamp, opts.figureFormat);
            filepath = fullfile(opts.outputDir, filename);
            try
                switch opts.figureFormat
                    case {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}
                        saveas(fig, filepath);
                    otherwise
                        exportgraphics(fig, filepath);
                end
                fprintf('    - 已保存: %s\n', filepath);
            catch ME
                fprintf('    ⚠ 保存失败 (%s): %s\n', labels{idx}, ME.message);
            end
        end
        if opts.closeAfterSave
            close(fig);
        end
    end
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
