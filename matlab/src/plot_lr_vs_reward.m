function plot_lr_vs_reward(results, varargin)
% plot_lr_vs_reward 一键绘制学习率动态调整与奖励曲线的可视化图表（增强版）
%
% 功能：
% - 从训练结果结构体 results 中提取 lr_trace([episode, actorLR, criticLR]) 与 episode_rewards，并进行有效性检查
% - 默认生成两个子图：子图1（Actor/Critic 学习率），子图2（Episode 奖励）
% - 可选：使用双 Y 轴（yyaxis）叠加显示学习率与奖励
% - 可选：移动平均奖励曲线（可配置窗口）
% - 可选：学习率对数坐标
% - 可选：分阶段标注（在特定 Episode 位置添加垂直线和文本）
% - 可选：将图像保存为 PNG 到指定目录
%
% 用法示例：
%   % 示例1：基本用法（训练完成后直接调用）
%   plot_lr_vs_reward(results);
%
%   % 示例2：使用双 Y 轴模式
%   plot_lr_vs_reward(results, 'dual_axis', true);
%
%   % 示例3：保存图片
%   plot_lr_vs_reward(results, 'save_fig', true, 'output_dir', 'project_document/figures/');
%
%   % 示例4：移动平均奖励 + 学习率对数坐标 + 阶段标注
%   plot_lr_vs_reward(results, 'ma_window', 10, 'log_lr', true, ...
%       'stage_positions', [50 100], 'stage_labels', {'warmup','anneal'});
%
% 输入参数（Name-Value）：
%   'dual_axis'       - 逻辑值，是否在同一坐标系使用双Y轴叠加显示（默认 false）
%   'save_fig'        - 逻辑值，是否保存图片到文件（默认 false）
%   'output_dir'      - 字符串，输出目录（默认 'project_document/figures/'）
%   'ma_window'       - 正整数，移动平均奖励窗口（默认 0 关闭；推荐 10）
%   'log_lr'          - 逻辑值，学习率使用对数坐标（默认 false）
%   'stage_positions' - 数值向量，要标注的 Episode 位置，默认 [] 关闭
%   'stage_labels'    - 元胞字符串数组，与 stage_positions 对应的标签，默认 {}
%   'stage_line_style'- 线型（默认 '--'）
%   'stage_line_width'- 线宽（默认 1.0）
%   'stage_color'     - 颜色（RGB 向量，默认 [0.3 0.3 0.3]）
%
% 兼容性：
% - 兼容 SAC/PPO/DDPG/DQN 四算法。
% - DQN 的 actorLR 为 NaN 时将自动忽略，不绘制 Actor 曲线，仅绘制 Critic（Q 网络）。
%
% 作者：Augment Agent（自动生成）
% -------------------------------------------------------------------------

% ------------------------- 参数解析与校验 -------------------------------
validateattributes(results, {'struct'}, {'nonempty'}, mfilename, 'results');

p = inputParser; p.FunctionName = mfilename;
addParameter(p, 'dual_axis',       false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'save_fig',        false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'output_dir',      "project_document/figures/", @(x) (ischar(x) || isstring(x)) && ~isempty(x));
addParameter(p, 'ma_window',       0,    @(x) isnumeric(x) && isscalar(x) && x>=0);
addParameter(p, 'log_lr',          false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'stage_positions', [],   @(x) isnumeric(x));
addParameter(p, 'stage_labels',    {},   @(x) iscell(x) || isstring(x));
addParameter(p, 'stage_line_style','--', @(x) ischar(x) || isstring(x));
addParameter(p, 'stage_line_width',1.0,  @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'stage_color',     [0.3 0.3 0.3], @(x) isnumeric(x) && (numel(x)==3));
parse(p, varargin{:});
opt = p.Results;
opt.dual_axis = logical(opt.dual_axis);
opt.save_fig  = logical(opt.save_fig);
opt.log_lr    = logical(opt.log_lr);
opt.output_dir = char(string(opt.output_dir));
if isstring(opt.stage_labels), opt.stage_labels = cellstr(opt.stage_labels); end

% --------------------------- 数据提取 ----------------------------------
hasLt = isfield(results, 'lr_trace') && ~isempty(results.lr_trace) && isnumeric(results.lr_trace);
hasEr = isfield(results, 'episode_rewards') && ~isempty(results.episode_rewards) && isnumeric(results.episode_rewards);

if ~hasLt && ~hasEr
    error('[plot_lr_vs_reward] results 中未找到 lr_trace 或 episode_rewards，无法绘图。');
end

lt = [];
if hasLt
    lt = results.lr_trace;
    % 形状检查：期望为 N×3（[episode, actorLR, criticLR]）
    if size(lt,2) < 2
        warning('[plot_lr_vs_reward] lr_trace 列数 < 2，无法解析学习率曲线。将忽略学习率，仅绘制奖励。');
        hasLt = false;
    end
end

er = [];
if hasEr
    er = results.episode_rewards(:);
end

% 尝试解析算法名（用于标题/文件命名，不强依赖）
algo = 'DRL';
try
    if isfield(results, 'algorithm') && ~isempty(results.algorithm)
        algo = char(string(results.algorithm));
    elseif isfield(results, 'algo_name') && ~isempty(results.algo_name)
        algo = char(string(results.algo_name));
    elseif isfield(results, 'agent_name') && ~isempty(results.agent_name)
        algo = char(string(results.agent_name));
    end
catch
end

% --------------------------- 绘图逻辑 ----------------------------------
figure('Name','学习率与奖励分析','Color','w');

if ~opt.dual_axis
    % 子图模式：上-学习率，下-奖励
    tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

    % 子图1：学习率
    ax1 = nexttile; hold(ax1,'on'); legends = {};
    if hasLt
        ep = lt(:,1);
        % 解析 actor/critic 列
        actorLR = []; criticLR = [];
        if size(lt,2) >= 3
            actorLR  = lt(:,2); criticLR = lt(:,3);
        else
            criticLR = lt(:,2);
        end
        % 绘制 Actor 学习率（若存在且非全 NaN）
        if ~isempty(actorLR) && any(isfinite(actorLR))
            plot(ax1, ep, actorLR, '-b', 'LineWidth', 1.2); legends{end+1} = 'actorLR';
        end
        % 绘制 Critic 学习率
        if ~isempty(criticLR) && any(isfinite(criticLR))
            plot(ax1, ep, criticLR, '-r', 'LineWidth', 1.2); legends{end+1} = 'criticLR';
        end
        if opt.log_lr, set(ax1,'YScale','log'); end
        % 阶段标注
        add_stage_annotations(ax1, opt, ep);
    else
        text(0.5, 0.5, '无学习率数据（lr_trace 缺失）', 'HorizontalAlignment','center','Parent',ax1);
    end
    grid(ax1,'on'); xlabel(ax1,'回合（Episode）'); ylabel(ax1,'学习率'); title(ax1,sprintf('%s 学习率变化（余弦退火）', algo));
    if ~isempty(legends), legend(ax1, legends, 'Location','best'); end

    % 子图2：奖励
    ax2 = nexttile; hold(ax2,'on');
    if hasEr
        plot(ax2, 1:numel(er), er, '-k', 'LineWidth', 1.0);
        % 移动平均奖励（可选）
        if opt.ma_window>=2
            er_ma = safe_movmean(er, opt.ma_window);
            plot(ax2, 1:numel(er_ma), er_ma, '-', 'Color', [0 0.6 0], 'LineWidth', 1.2);
            legend(ax2, {'reward','reward(movmean)'}, 'Location','best');
        end
        grid(ax2,'on'); xlabel(ax2,'回合（Episode）'); ylabel(ax2,'回合奖励'); title(ax2,sprintf('%s Episode Reward', algo));
        % 阶段标注
        add_stage_annotations(ax2, opt, 1:numel(er));
    else
        text(0.5, 0.5, '无奖励数据（episode_rewards 缺失）', 'HorizontalAlignment','center','Parent',ax2);
        grid(ax2,'on'); xlabel(ax2,'回合（Episode）'); ylabel(ax2,'回合奖励'); title(ax2,sprintf('%s Episode Reward', algo));
    end
else
    % 双 Y 轴模式：同一坐标系叠加 LR 与 Reward
    ax = gca; hold(ax,'on'); legendsL = {}; legendsR = {};
    if hasLt
        ep = lt(:,1); actorLR = []; criticLR = [];
        if size(lt,2) >= 3, actorLR  = lt(:,2); criticLR = lt(:,3); else, criticLR = lt(:,2); end
        yyaxis left;
        if ~isempty(actorLR) && any(isfinite(actorLR))
            plot(ep, actorLR, '-b', 'LineWidth', 1.2); legendsL{end+1} = 'actorLR';
        end
        if ~isempty(criticLR) && any(isfinite(criticLR))
            plot(ep, criticLR, '-r', 'LineWidth', 1.2); legendsL{end+1} = 'criticLR';
        end
        ylabel('学习率'); if opt.log_lr, set(gca,'YScale','log'); end
        % 阶段标注（基于 episode 轴）
        add_stage_annotations(ax, opt, ep);
    end
    if hasEr
        yyaxis right;
        plot(1:numel(er), er, '-k', 'LineWidth', 1.0); legendsR{end+1} = 'reward'; ylabel('回合奖励');
        if opt.ma_window>=2
            er_ma = safe_movmean(er, opt.ma_window);
            hold on; plot(1:numel(er_ma), er_ma, '-', 'Color', [0 0.6 0], 'LineWidth', 1.2); legendsR{end+1} = 'reward(movmean)';
        end
    end
    grid(ax,'on'); xlabel(ax,'回合（Episode）'); title(ax,sprintf('%s 学习率与奖励（双Y轴对比）', algo));
    legends = [legendsL, legendsR]; if ~isempty(legends), legend(ax, legends, 'Location','best'); end
end

% --------------------------- 保存图片 ----------------------------------
if opt.save_fig
    try
        if ~exist(opt.output_dir, 'dir'), mkdir(opt.output_dir); end
        ts = datestr(now, 'yyyymmdd_HHMMSS');
        safeAlgo = regexprep(lower(algo), '[^a-z0-9_\-]', '_');
        outpath = fullfile(opt.output_dir, sprintf('lr_vs_reward_%s_%s.png', safeAlgo, ts));
        try
            exportgraphics(gcf, outpath, 'Resolution', 150);
        catch
            saveas(gcf, outpath);
        end
        fprintf('[plot_lr_vs_reward] 图像已保存到: %s\n', outpath);
    catch ME
        % 使用错误标识符，便于日志过滤
        warning(ME.identifier, '[plot_lr_vs_reward] 保存图片失败：%s', ME.message);
    end
end

end % function

% ============================== 局部函数 =================================
function y = safe_movmean(x, w)
% safe_movmean 兼容性移动平均：优先使用 movmean，不可用时退化为卷积
try
    y = movmean(x, w);
catch
    y = conv(x(:), ones(w,1)/w, 'same');
end
end

function add_stage_annotations(ax, opt, xAxis)
% add_stage_annotations 在坐标轴 ax 上添加阶段竖线和文本标注
if isempty(opt.stage_positions), return; end
sp = opt.stage_positions(:)';
sl = opt.stage_labels; if ~iscell(sl), sl = cellstr(sl); end
hold(ax,'on');
% 为避免超出范围，仅对在当前 x 范围内的 position 进行绘制
xlim_curr = get(ax,'XLim');
ylim_curr = get(ax,'YLim');
for i = 1:numel(sp)
    ep = sp(i);
    if ep < min(xAxis) || ep > max(xAxis), continue; end
    xline(ax, ep, 'Color', opt.stage_color, 'LineStyle', char(opt.stage_line_style), 'LineWidth', opt.stage_line_width);
    if i <= numel(sl) && ~isempty(sl{i})
        text(ax, ep, ylim_curr(2) - 0.05*(ylim_curr(2)-ylim_curr(1)), char(sl{i}), ...
            'Color', opt.stage_color, 'HorizontalAlignment','right', 'VerticalAlignment','top', 'Interpreter','none');
    end
end
end

