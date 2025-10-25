% plot_best_episode.m
% 用途：加载 results/.../best_episode_data.mat 并以论文级排版生成 2x2 关键曲线图
% 输入：无（脚本）。如需自定义数据路径/算法名称，请修改“用户可配置区”。
% 输出：results/best_run/best_episode_analysis.png（1200x900, 300DPI）
% 兼容性：
% - 字段别名：Battery_Power 或 P_batt；SOH_Diff 或 SOH_diff
% - 时间单位：默认假设 Simulink 日志时间为秒，统一换算为小时
% - 兼容不同算法目录：默认使用 results/best_run/；若不存在可手动改 dataPath

%% 用户可配置区
algName  = 'AUTO'; % 可填写 'SAC'/'PPO'/'DDPG' 等；'AUTO' 仅用于标题显示
% 默认数据路径（如需画其它算法，改为 results/<ALG>/best_run/best_episode_data.mat）
dataPath = fullfile(pwd,'results','best_run','best_episode_data.mat');
if ~isfile(dataPath)
    % 常见备选：results/<ALG>/best_run/
    candidates = { ...
        fullfile(pwd,'results','SAC','best_run','best_episode_data.mat'), ...
        fullfile(pwd,'results','PPO','best_run','best_episode_data.mat'), ...
        fullfile(pwd,'results','DDPG','best_run','best_episode_data.mat') ...
    };
    for ii = 1:numel(candidates)
        if isfile(candidates{ii}), dataPath = candidates{ii}; break; end
    end
end

outDir   = fullfile(pwd,'results','best_run');
if ~exist(outDir,'dir'), mkdir(outDir); end
outPng   = fullfile(outDir,'best_episode_analysis.png');
outPdf   = fullfile(outDir,'best_episode_analysis.pdf');


%% 载入数据并做字段检查
S = load(dataPath);

% 字段解析（含别名）
SOC = must_get_ts(S, {'Battery_SOC','battery_soc'}, 'Battery_SOC');
SOH = must_get_ts(S, {'Battery_SOH','battery_soh'}, 'Battery_SOH');
PWR = must_get_ts(S, {'Battery_Power','P_batt','P_Batt','P_battery'}, 'Battery_Power');
COST= must_get_ts(S, {'TotalCost','total_cost'}, 'TotalCost');

% 时间向统一“小时”
[t_soc, soc]   = ts_to_hours_and_data(SOC);
[~, soh]       = ts_to_hours_and_data(SOH);
[t_pwr, pwr]   = ts_to_hours_and_data(PWR);
[t_cost, cost] = ts_to_hours_and_data(COST);
% 对齐时间轴（若必要，简单对齐到最短长度与公共区间）
[t_min, t_max] = deal(max([t_soc(1), t_pwr(1), t_cost(1)]), min([t_soc(end), t_pwr(end), t_cost(end)]));
SOC_mask  = (t_soc>=t_min & t_soc<=t_max);
SOH_mask  = (t_soc>=t_min & t_soc<=t_max); % 假设 SOC/ SOH 同步
PWR_mask  = (t_pwr>=t_min & t_pwr<=t_max);
COST_mask = (t_cost>=t_min & t_cost<=t_max);

% 单位与量纲
soc_pct  = soc*100;         % [0,1] → %
soh_pct  = soh*100;         % [0,1] → %（若已是%，数值应在0-100区间）
% Battery power → kW（若数据量级显著大于数kW，按 W→kW 兜底转换）
if max(abs(pwr)) > 200 % 粗略阈值，>200 视为单位为 W
    pwr_kw = pwr/1000;
else
    pwr_kw = pwr;
end

%% 论文级排版（颜色/字号/网格）
fig = figure('Color','w','Position',[100 100 1200 900]);
% 自定义颜色（避免默认蓝）
C_RED   = [0.80 0.20 0.20];
C_BLUE  = [0.20 0.40 0.80];
C_GREEN = [0.20 0.60 0.30];
C_ORAN  = [0.90 0.50 0.10];
LW = 1.8;

% 统一默认字体（对本图有效）
set(gca,'FontName','Times New Roman'); %#ok<SAGROW>

% 子图 1：SOC
ax1 = subplot(2,2,1); hold(ax1,'on'); grid(ax1,'on');
set(ax1,'GridLineStyle',':','GridAlpha',0.3,'FontSize',10,'FontName','Times New Roman');
plot(ax1, t_soc(SOC_mask), soc_pct(SOC_mask), '-', 'Color', C_RED, 'LineWidth', LW);
xlabel(ax1,'Time (hours)','FontSize',12);
ylabel(ax1,'SOC (%)','FontSize',12);
title(ax1,'Battery State of Charge (SOC)','FontSize',14);
ylim(ax1,[0 100]);

% 子图 2：SOH
ax2 = subplot(2,2,2); hold(ax2,'on'); grid(ax2,'on');
set(ax2,'GridLineStyle',':','GridAlpha',0.3,'FontSize',10,'FontName','Times New Roman');
plot(ax2, t_soc(SOH_mask), soh_pct(SOH_mask), '-', 'Color', C_BLUE, 'LineWidth', LW);
xlabel(ax2,'Time (hours)','FontSize',12);
ylabel(ax2,'SOH (%)','FontSize',12);
title(ax2,'Battery State of Health (SOH)','FontSize',14);
ylim(ax2,[max(0,min(95,min(soh_pct))-1) 100]);

% 子图 3：Battery Power
ax3 = subplot(2,2,3); hold(ax3,'on'); grid(ax3,'on');
set(ax3,'GridLineStyle',':','GridAlpha',0.3,'FontSize',10,'FontName','Times New Roman');
plot(ax3, t_pwr(PWR_mask), pwr_kw(PWR_mask), '-', 'Color', C_GREEN, 'LineWidth', LW);
yline(ax3,0,'k-');
xlabel(ax3,'Time (hours)','FontSize',12);
ylabel(ax3,'Power (kW)','FontSize',12);
title(ax3,'Battery Charge/Discharge Power','FontSize',14);

% 子图 4：Total Cost（累计）
ax4 = subplot(2,2,4); hold(ax4,'on'); grid(ax4,'on');
set(ax4,'GridLineStyle',':','GridAlpha',0.3,'FontSize',10,'FontName','Times New Roman');
plot(ax4, t_cost(COST_mask), cost(COST_mask), '-', 'Color', C_ORAN, 'LineWidth', LW);
xlabel(ax4,'Time (hours)','FontSize',12);
ylabel(ax4,'Cumulative Cost','FontSize',12);
title(ax4,'Total Cost','FontSize',14);

% 总标题
try
    if strcmpi(algName,'AUTO')
        sgtitle(sprintf('Best Episode — %s', strrep(dataPath(numel(pwd)+2:end), filesep, '/')), 'FontSize',14, 'FontWeight','bold');
    else
        sgtitle(sprintf('Best Episode — Algorithm: %s', algName), 'FontSize',14, 'FontWeight','bold');
    end
catch
    sgtitle('Best Episode', 'FontSize',14, 'FontWeight','bold');
end

% 导出 PNG（300 DPI）
try
    print(fig, outPng, '-dpng', '-r300');
    fprintf('✓ 已生成分析图：%s\n', outPng);
catch ME
    warning(ME.identifier, '导出 PNG 失败：%s', ME.message);
end

% 导出 PDF（矢量）
try
    exportgraphics(fig, outPdf, 'ContentType','vector', 'BackgroundColor','white');
    fprintf('✓ 已生成分析图（PDF）：%s\n', outPdf);
catch ME
    warning(ME.identifier, '导出 PDF 失败：%s', ME.message);
end

% 生成单指标大图（800x600，PNG+PDF）
create_single_fig(fullfile(outDir,'SOC_vs_time'), ...
    'Battery State of Charge (SOC) over 30 Days', 'Time (hours)', 'SOC (%)', ...
    t_soc(SOC_mask), soc_pct(SOC_mask), [0.80 0.20 0.20]);
create_single_fig(fullfile(outDir,'SOH_vs_time'), ...
    'Battery State of Health (SOH) over 30 Days', 'Time (hours)', 'SOH (%)', ...
    t_soc(SOH_mask), soh_pct(SOH_mask), [0.20 0.40 0.80]);
create_single_fig(fullfile(outDir,'Power_vs_time'), ...
    'Battery Charge/Discharge Power over 30 Days', 'Time (hours)', 'Power (kW)', ...
    t_pwr(PWR_mask), pwr_kw(PWR_mask), [0.20 0.60 0.30]);
create_single_fig(fullfile(outDir,'TotalCost_vs_time'), ...
    'Total Cost over 30 Days', 'Time (hours)', 'Cumulative Cost', ...
    t_cost(COST_mask), cost(COST_mask), [0.90 0.50 0.10]);

%% 辅助函数：字段抓取 + 别名支持
function ts = must_get_ts(S, keys, dispName)
    for i = 1:numel(keys)
        k = keys{i};
        if isfield(S, k)

            v = S.(k);
            if isa(v,'timeseries')
                ts = v; return;
            else
                error('plot_best_episode:Type','字段 %s 存在但类型为 %s，应为 timeseries。', k, class(v));
            end
        end
    end
    error('plot_best_episode:Missing','缺少必需字段 %s（支持别名：%s）', dispName, strjoin(keys,', '));
end

%% 辅助函数：timeseries → (t[h], data)
function [t_h, y] = ts_to_hours_and_data(ts)
    t = ts.Time;
    if isdatetime(t)
        t_h = hours(t - t(1));
    else
        % Simulink 日志默认秒 → 小时
        t_h = (t - t(1)) / 3600;
    end
    y = ts.Data;
end



%% 辅助函数：生成单指标大图（PNG+PDF）
function create_single_fig(basePath, titleText, xLabelText, yLabelText, t, y, color)
    fig = figure('Color','w','Position',[100 100 800 600]);
    ax = axes(fig); hold(ax,'on'); grid(ax,'on');
    set(ax,'GridLineStyle',':','GridAlpha',0.3,'FontSize',10,'FontName','Times New Roman');
    plot(ax, t, y, '-', 'Color', color, 'LineWidth', 1.8);
    xlabel(ax, xLabelText, 'FontSize',12, 'FontName','Times New Roman');
    ylabel(ax, yLabelText, 'FontSize',12, 'FontName','Times New Roman');
    title(ax, titleText, 'FontSize',14, 'FontName','Times New Roman');
    pngPath = [basePath '.png'];
    pdfPath = [basePath '.pdf'];
    try
        print(fig, pngPath, '-dpng', '-r300');
        exportgraphics(fig, pdfPath, 'ContentType','vector', 'BackgroundColor','white');
        fprintf('✓ 生成：%s, %s\n', pngPath, pdfPath);
    catch ME
        warning(ME.identifier, '导出单图失败：%s', ME.message);
    end
    close(fig);
end
