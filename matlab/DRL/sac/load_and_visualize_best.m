function load_and_visualize_best(options)
%LOAD_AND_VISUALIZE_BEST 加载并可视化最优训练结果
%   从best_run目录加载保存的最优智能体和episode数据,并生成可视化图表
%
% 用法:
%   load_and_visualize_best()  - 使用默认选项
%   load_and_visualize_best(options)  - 自定义可视化选项
%
% 选项:
%   options.showFigures - 是否显示图表窗口 (默认true)
%   options.saveFigures - 是否保存图表文件 (默认true)
%   options.outputDir - 图表保存目录 (默认best_run/figures)
%   options.generateReport - 是否生成文本报告 (默认true)
%
% 示例:
%   % 基本用法
%   load_and_visualize_best();
%   
%   % 只查看不保存
%   load_and_visualize_best(struct('saveFigures', false));
%   
%   % 自定义输出目录
%   opts = struct('outputDir', 'my_results');
%   load_and_visualize_best(opts);

    if nargin < 1
        options = struct();
    end
    
    % 设置默认选项
    opts = parse_options(options);
    
    fprintf('\n========================================\n');
    fprintf('  最优训练结果可视化系统\n');
    fprintf('========================================\n');
    fprintf('启动时间: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    
    % 1. 加载最优结果
    fprintf('=== 步骤1: 加载最优结果 ===\n');
    [agent, episodeData, meta] = run_best_manager('load');
    
    % 检查是否成功加载
    if isempty(agent)
        fprintf('\n✗ 未找到最优智能体\n');
        fprintf('  请先运行训练: main() 或 train_model()\n');
        return;
    end
    
    if isempty(meta) || ~isfield(meta, 'reward')
        fprintf('\n⚠ 未找到元数据,可能是首次运行\n');
    else
        display_meta_info(meta);
    end
    
    % 2. 准备可视化数据
    fprintf('\n=== 步骤2: 准备可视化数据 ===\n');
    
    % 将episode数据推送到workspace供visualization使用
    if ~isempty(episodeData)
        push_data_to_workspace(episodeData);
    else
        fprintf('⚠ 无episode数据,将只显示训练统计\n');
    end
    
    % 准备训练结果数据
    trainingResults = [];
    if ~isempty(meta) && isfield(meta, 'results_summary')
        trainingResults = meta.results_summary;
    end
    
    % 3. 生成可视化
    fprintf('\n=== 步骤3: 生成可视化图表 ===\n');
    
    try
        % 添加visualization.m路径
        project_root = resolve_project_root();
        matlab_src = fullfile(project_root, 'matlab', 'src');
        if exist(matlab_src, 'dir')
            addpath(matlab_src);
        end
        run_manager_dir = fullfile(matlab_src, 'run_manager');
        if exist(run_manager_dir, 'dir')
            addpath(run_manager_dir);
        end
        
        % 调用visualization函数
        viz_opts = struct();
        viz_opts.workspace = 'base';
        viz_opts.showFigures = opts.showFigures;
        viz_opts.saveFigures = opts.saveFigures;
        viz_opts.outputDir = opts.outputDir;
        viz_opts.filePrefix = 'Best';
        viz_opts.figureFormat = 'png';
        viz_opts.closeAfterSave = ~opts.showFigures;
        viz_opts.timestamp = datestr(now, 'yyyyMMdd_HHmmss');
        
        if ~isempty(trainingResults)
            viz_opts.trainingResults = trainingResults;
        end
        
        % 检查visualization函数是否可用
        if exist('visualization', 'file') ~= 2
            fprintf('⚠ 未找到visualization函数\n');
            fprintf('  将使用基础绘图功能\n');
            create_basic_plots(episodeData, meta, opts);
        else
            visualization(viz_opts);
            fprintf('✓ 高级可视化完成\n');
        end
        
    catch ME
        fprintf('✗ 可视化失败: %s\n', ME.message);
        fprintf('  尝试使用基础绘图...\n');
        try
            create_basic_plots(episodeData, meta, opts);
        catch ME2
            fprintf('✗ 基础绘图也失败: %s\n', ME2.message);
        end
    end
    
    % 4. 生成文本报告
    if opts.generateReport
        fprintf('\n=== 步骤4: 生成文本报告 ===\n');
        generate_text_report(agent, episodeData, meta, opts);
    end
    
    fprintf('\n========================================\n');
    fprintf('  可视化完成\n');
    fprintf('========================================\n');
    if opts.saveFigures
        fprintf('图表已保存至: %s\n', opts.outputDir);
    end
    fprintf('\n');
end

%% ========================================================================
%% 核心功能函数
%% ========================================================================

function opts = parse_options(options)
    %PARSE_OPTIONS 解析和设置默认选项
    
    % 获取默认输出目录
    project_root = resolve_project_root();
    default_output = fullfile(project_root, 'results', 'best_run', 'figures');
    
    opts = struct();
    opts.showFigures = get_option(options, 'showFigures', true);
    opts.saveFigures = get_option(options, 'saveFigures', true);
    opts.outputDir = get_option(options, 'outputDir', default_output);
    opts.generateReport = get_option(options, 'generateReport', true);
    
    % 创建输出目录
    if opts.saveFigures && ~exist(opts.outputDir, 'dir')
        mkdir(opts.outputDir);
    end
end

function display_meta_info(meta)
    %DISPLAY_META_INFO 显示元数据信息
    
    fprintf('\n最优训练记录:\n');
    fprintf('  🏆 最佳奖励: %.2f\n', meta.reward);
    fprintf('  📅 训练时间: %s\n', meta.timestamp);
    fprintf('  🔄 训练回合: %d episodes\n', meta.episodes);
    
    if isfield(meta, 'training_time') && meta.training_time > 0
        hours = floor(meta.training_time / 3600);
        minutes = floor(mod(meta.training_time, 3600) / 60);
        fprintf('  ⏱ 训练耗时: %d小时%d分钟\n', hours, minutes);
    end
    
    if isfield(meta, 'average_reward')
        fprintf('  📊 平均奖励: %.2f\n', meta.average_reward);
    end
end

function push_data_to_workspace(episodeData)
    %PUSH_DATA_TO_WORKSPACE 将数据推送到base workspace
    
    fields = fieldnames(episodeData);
    pushedCount = 0;
    
    for i = 1:length(fields)
        fieldName = fields{i};
        fieldValue = episodeData.(fieldName);
        
        % 只推送有用的变量
        if contains(fieldName, {'Battery', 'SOC', 'SOH', 'Power', 'Cost', 'Reward'})
            try
                assignin('base', fieldName, fieldValue);
                pushedCount = pushedCount + 1;
            catch
                % 忽略推送失败
            end
        end
    end
    
    fprintf('✓ 已推送 %d 个变量到workspace\n', pushedCount);
end

function create_basic_plots(episodeData, meta, opts)
    %CREATE_BASIC_PLOTS 创建基础可视化图表(当visualization不可用时)
    
    if isempty(episodeData) || ~isfield(episodeData, 'Battery_SOC')
        fprintf('⚠ 无法创建图表:缺少电池数据\n');
        return;
    end
    
    % 创建主图表
    fig = figure('Name', '最优训练结果', 'Position', [100, 100, 1200, 800], ...
        'Color', 'w', 'Visible', ternary(opts.showFigures, 'on', 'off'));
    
    % 子图1: SOC时序
    subplot(2, 2, 1);
    plot_battery_soc(episodeData.Battery_SOC);
    
    % 子图2: SOH时序
    subplot(2, 2, 2);
    if isfield(episodeData, 'Battery_SOH')
        plot_battery_soh(episodeData.Battery_SOH);
    else
        text(0.5, 0.5, '无SOH数据', 'HorizontalAlignment', 'center');
        axis off;
    end
    
    % 子图3: 功率时序
    subplot(2, 2, 3);
    if isfield(episodeData, 'Battery_Power')
        plot_battery_power(episodeData.Battery_Power);
    else
        text(0.5, 0.5, '无功率数据', 'HorizontalAlignment', 'center');
        axis off;
    end
    
    % 子图4: 统计信息
    subplot(2, 2, 4);
    plot_statistics(episodeData, meta);
    
    sgtitle('最优训练结果 - 电池性能分析', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 保存图表
    if opts.saveFigures
        filename = fullfile(opts.outputDir, sprintf('Best_basic_%s.png', datestr(now, 'yyyyMMdd_HHmmss')));
        saveas(fig, filename);
        fprintf('✓ 基础图表已保存: %s\n', filename);
    end
    
    if ~opts.showFigures
        close(fig);
    end
end

function plot_battery_soc(soc_ts)
    %PLOT_BATTERY_SOC 绘制SOC时序图
    
    if isa(soc_ts, 'timeseries')
        time_hours = soc_ts.Time / 3600;
        soc_data = soc_ts.Data;
    else
        time_hours = 1:length(soc_ts);
        soc_data = soc_ts;
    end
    
    % 转换为百分比
    if max(soc_data) <= 1
        soc_data = soc_data * 100;
    end
    
    plot(time_hours, soc_data, 'b-', 'LineWidth', 2);
    xlabel('时间 (小时)');
    ylabel('SOC (%)');
    title('电池SOC');
    grid on;
    ylim([0, 100]);
end

function plot_battery_soh(soh_ts)
    %PLOT_BATTERY_SOH 绘制SOH时序图
    
    if isa(soh_ts, 'timeseries')
        time_hours = soh_ts.Time / 3600;
        soh_data = soh_ts.Data;
    else
        time_hours = 1:length(soh_ts);
        soh_data = soh_ts;
    end
    
    % 转换为百分比
    if max(soh_data) <= 1
        soh_data = soh_data * 100;
    end
    
    plot(time_hours, soh_data, 'g-', 'LineWidth', 2);
    xlabel('时间 (小时)');
    ylabel('SOH (%)');
    title('电池SOH');
    grid on;
    ylim([90, 100]);
end

function plot_battery_power(power_ts)
    %PLOT_BATTERY_POWER 绘制功率时序图
    
    if isa(power_ts, 'timeseries')
        time_hours = power_ts.Time / 3600;
        power_data = power_ts.Data / 1000;  % 转换为kW
    else
        time_hours = 1:length(power_ts);
        power_data = power_ts / 1000;
    end
    
    % 分别绘制充电和放电
    positive = max(0, power_data);
    negative = min(0, power_data);
    
    area(time_hours, positive, 'FaceColor', 'r', 'FaceAlpha', 0.5);
    hold on;
    area(time_hours, negative, 'FaceColor', 'b', 'FaceAlpha', 0.5);
    hold off;
    
    xlabel('时间 (小时)');
    ylabel('功率 (kW)');
    title('电池充放电功率');
    legend('放电', '充电');
    grid on;
end

function plot_statistics(episodeData, meta)
    %PLOT_STATISTICS 显示统计信息
    
    axis off;
    
    stats_text = {'=== 最优结果统计 ==='};
    
    if ~isempty(meta) && isfield(meta, 'reward')
        stats_text{end+1} = sprintf('最佳奖励: %.2f', meta.reward);
        stats_text{end+1} = sprintf('训练时间: %s', meta.timestamp);
    end
    
    if isfield(episodeData, 'Battery_SOC')
        soc = episodeData.Battery_SOC.Data;
        if max(soc) <= 1, soc = soc * 100; end
        stats_text{end+1} = '';
        stats_text{end+1} = sprintf('SOC 平均: %.1f%%', mean(soc));
        stats_text{end+1} = sprintf('SOC 范围: %.1f%% - %.1f%%', min(soc), max(soc));
    end
    
    if isfield(episodeData, 'Battery_SOH')
        soh = episodeData.Battery_SOH.Data;
        if max(soh) <= 1, soh = soh * 100; end
        stats_text{end+1} = '';
        stats_text{end+1} = sprintf('SOH 平均: %.1f%%', mean(soh));
        stats_text{end+1} = sprintf('SOH 衰减: %.2f%%', 100 - min(soh));
    end
    
    if isfield(episodeData, 'cumulative_cost')
        stats_text{end+1} = '';
        stats_text{end+1} = sprintf('累计成本: $%.2f', episodeData.cumulative_cost);
    end
    
    text(0.1, 0.5, stats_text, 'VerticalAlignment', 'middle', 'FontSize', 10, ...
        'FontName', 'FixedWidth');
end

function generate_text_report(agent, episodeData, meta, opts)
    %GENERATE_TEXT_REPORT 生成文本格式的分析报告
    
    reportPath = fullfile(opts.outputDir, sprintf('Best_report_%s.txt', datestr(now, 'yyyyMMdd_HHmmss')));
    
    try
        fid = fopen(reportPath, 'w');
        
        fprintf(fid, '===============================================\n');
        fprintf(fid, '     SAC最优训练结果分析报告\n');
        fprintf(fid, '===============================================\n\n');
        fprintf(fid, '生成时间: %s\n\n', datestr(now));
        
        % 元数据部分
        if ~isempty(meta)
            fprintf(fid, '--- 训练元数据 ---\n');
            fprintf(fid, '最佳奖励: %.2f\n', meta.reward);
            fprintf(fid, '训练时间: %s\n', meta.timestamp);
            fprintf(fid, '训练回合: %d episodes\n', meta.episodes);
            if isfield(meta, 'training_time')
                fprintf(fid, '训练耗时: %.1f 分钟\n', meta.training_time / 60);
            end
            fprintf(fid, '\n');
        end
        
        % 智能体信息
        if ~isempty(agent)
            fprintf(fid, '--- 智能体信息 ---\n');
            fprintf(fid, '类型: %s\n', class(agent));
            fprintf(fid, '观测维度: %s\n', mat2str(agent.ObservationInfo.Dimension));
            fprintf(fid, '动作维度: %s\n', mat2str(agent.ActionInfo.Dimension));
            fprintf(fid, '\n');
        end
        
        % Episode数据统计
        if ~isempty(episodeData)
            fprintf(fid, '--- Episode数据统计 ---\n');
            
            if isfield(episodeData, 'Battery_SOC')
                soc = episodeData.Battery_SOC.Data;
                if max(soc) <= 1, soc = soc * 100; end
                fprintf(fid, 'Battery SOC:\n');
                fprintf(fid, '  平均值: %.1f%%\n', mean(soc));
                fprintf(fid, '  范围: %.1f%% - %.1f%%\n', min(soc), max(soc));
                fprintf(fid, '  标准差: %.1f%%\n', std(soc));
            end
            
            if isfield(episodeData, 'Battery_SOH')
                soh = episodeData.Battery_SOH.Data;
                if max(soh) <= 1, soh = soh * 100; end
                fprintf(fid, 'Battery SOH:\n');
                fprintf(fid, '  平均值: %.1f%%\n', mean(soh));
                fprintf(fid, '  最终值: %.1f%%\n', soh(end));
                fprintf(fid, '  总衰减: %.2f%%\n', 100 - min(soh));
            end
            
            if isfield(episodeData, 'cumulative_cost')
                fprintf(fid, '经济指标:\n');
                fprintf(fid, '  累计成本: $%.2f\n', episodeData.cumulative_cost);
            end
        end
        
        fprintf(fid, '\n===============================================\n');
        fprintf(fid, '报告结束\n');
        fprintf(fid, '===============================================\n');
        
        fclose(fid);
        fprintf('✓ 文本报告已保存: %s\n', reportPath);
        
    catch ME
        fprintf('⚠ 文本报告生成失败: %s\n', ME.message);
        if exist('fid', 'var') && fid ~= -1
            fclose(fid);
        end
    end
end

%% ========================================================================
%% 辅助函数
%% ========================================================================

function value = get_option(options, field, default_value)
    %GET_OPTION 获取选项值或使用默认值
    
    if isstruct(options) && isfield(options, field)
        value = options.(field);
    else
        value = default_value;
    end
end

function out = ternary(condition, true_val, false_val)
    %TERNARY 三元运算符
    
    if condition
        out = true_val;
    else
        out = false_val;
    end
end

function project_root = resolve_project_root()
    persistent cached_root
    if isempty(cached_root)
        current_dir = fileparts(mfilename('fullpath'));
        cached_root = current_dir;
        max_depth = 10;
        for i = 1:max_depth
            if exist(fullfile(cached_root, 'matlab'), 'dir') && exist(fullfile(cached_root, 'model'), 'dir')
                break;
            end
            parent_dir = fileparts(cached_root);
            if isempty(parent_dir) || strcmp(parent_dir, cached_root)
                error('SAC:ProjectRootNotFound', '无法从路径%s定位项目根目录', current_dir);
            end
            cached_root = parent_dir;
        end
        if ~(exist(fullfile(cached_root, 'matlab'), 'dir') && exist(fullfile(cached_root, 'model'), 'dir'))
            error('SAC:ProjectRootNotFound', '无法从路径%s定位项目根目录', current_dir);
        end
    end
    project_root = cached_root;
end

