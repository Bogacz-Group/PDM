clc;
clear;

% Read raw data from Excel
raw = readtable('Triplet_Raw_Data.xlsx', 'Sheet', 'Sheet1', ...
                'Range', 'A3:P14', 'ReadVariableNames', false);

% Extract x columns (STDP ratio values) for each condition
% Columns: A=OR, C=RO, E=ORO-10, G=ORO-5, I=ORO(-5,+15), K=ROR-10, M=ROR-5, O=ROR(+5,-15)
OR_vals    = raw{:,1};   % column A
RO_vals    = raw{:,3};   % column C
ORO10_vals = raw{:,5};   % column E
ORO5_vals  = raw{:,7};   % column G
ORO515_vals  = raw{:,9};   % column I
ROR10_vals = raw{:,11};   % column K
ROR5_vals  = raw{:,13};  % column M
ROR515_vals  = raw{:,15};  % column O

% Remove NaNs and convert STDP ratio to % change
OR_pts    = (OR_vals(~isnan(OR_vals))    - 1) * 100;
RO_pts    = (RO_vals(~isnan(RO_vals))    - 1) * 100;
ORO10_pts = (ORO10_vals(~isnan(ORO10_vals)) - 1) * 100;
ORO5_pts  = (ORO5_vals(~isnan(ORO5_vals))  - 1) * 100;
ORO515_pts  = (ORO515_vals(~isnan(ORO515_vals))  - 1) * 100;
ROR10_pts = (ROR10_vals(~isnan(ROR10_vals)) - 1) * 100;
ROR5_pts  = (ROR5_vals(~isnan(ROR5_vals))  - 1) * 100;
ROR515_pts  = (ROR515_vals(~isnan(ROR515_vals))  - 1) * 100;

% Compute means and SEMs from individual datapoints
all_pts = {OR_pts, RO_pts, ORO10_pts, ORO5_pts, ORO515_pts, ROR10_pts, ROR5_pts, ROR515_pts};

targets_mean = cellfun(@mean, all_pts);
targets_sem  = cellfun(@(x) std(x)/sqrt(numel(x)), all_pts);

% Use targets_mean as your fitting targets
targets = targets_mean;

% % Target values (% change)
% targets ~ [-18.5, +25, +25, +27, +14, +2.5, 0, +16];
% Parameter bounds: [lambda, epsilon, w0, tau, alpha]
lb = [0.01,  -0.5, 0.1, 0.0,  0.0001];
ub = [0.5,   0.5, 0.9, 10.0, 2.0   ];

% Fixed simulation parameters
dt      = 0.01;
simTime = 200;
T       = round(simTime/dt);
t_anchor= 100;

% Gating function
g = @(x, eps) 1 ./ (1 + exp(-200*(x - eps)));

% Objective function
obj = @(p) compute_residuals(p, targets, dt, T, t_anchor, g);

% Multiple random starts to avoid local minima
n_starts = 50;
best_fval = inf;
best_p    = [];

% rng(61); % Uncomment to get reproducible initialization
for k = 1:n_starts
    p0 = lb + rand(size(lb)).*(ub - lb);
    [p_opt, fval] = fmincon(obj, p0, [], [], [], [], lb, ub, [], ...
                            optimoptions('fmincon', ...
                            'Display',              'off', ...
                            'MaxFunctionEvaluations', 10000, ...
                            'TolFun',               1e-8, ...
                            'TolX',                 1e-8));
    if fval < best_fval
        best_fval = fval;
        best_p    = p_opt;
    end
end

fprintf('Best parameters:\n');
fprintf('  lambda  = %.4f\n', best_p(1));
fprintf('  epsilon = %.4f\n', best_p(2));
fprintf('  w0      = %.4f\n', best_p(3));
fprintf('  tau     = %.4f ms\n', best_p(4));
fprintf('  alpha     = %.4f \n', best_p(5));
fprintf('Best SSE: %.4f\n', best_fval);

% Evaluate the model at the best-fit parameters
[~, pred] = compute_residuals(best_p, targets, dt, T, t_anchor, g);

fprintf('\nCondition       Target   Predicted\n');
cond_names = {'OR(-10)', 'RO(+10)', 'ORO(-10,+10)', 'ORO(-5,+5)', 'ORO(-5,+15)', ...
              'ROR(+10,-10)',  'ROR(+5,-5)',   'ROR(+5,-15)'};
for k = 1:8
    fprintf('  %-12s  %+6.1f   %+6.1f\n', cond_names{k}, targets(k), pred(k));
end

% --------------------------------------------------------
%% plotting

set(groot, 'defaultAxesFontName', 'Arial', ...
           'defaultTextInterpreter', 'tex');

% Colours per condition group
colors = [
    0.2 0.2 0.8;   % OR  — single pair depression
    0.2 0.2 0.8;   % RO  — single pair potentiation
    0.8 0.2 0.2;   % ORO-10
    0.8 0.2 0.2;   % ORO-5
    0.8 0.2 0.2;   % ORO(-5,+15)
    0.1 0.6 0.1;   % ROR-10
    0.1 0.6 0.1;   % ROR-5
    0.1 0.6 0.1;   % ROR(+5,-15)
];

% ---- Figure 1: Grouped bar chart ----
figure('Position', [100 100 950 450]);
b = bar([targets; pred]', 'grouped');
b(1).FaceColor = [0.2 0.4 0.8];
b(2).FaceColor = [0.9 0.4 0.1];
b(1).EdgeColor = [1.0 1.0 1.0];
b(2).EdgeColor = [1.0 1.0 1.0];
b(1).LineWidth = 1;
b(2).LineWidth = 1;
b(1).FaceAlpha = 0.65;
b(2).FaceAlpha = 0.85;

yline(0, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
hold on;

x_data = b(1).XEndPoints;   % x centres of the data bars

% Annotate residuals above predicted bars
% x_pos = b(2).XEndPoints;
% for k = 1:8
%     diff_val = pred(k) - targets(k);
%     y_offset = pred(k) + sign(pred(k)) * 1.8;
%     text(x_pos(k), y_offset, sprintf('%+.1f', diff_val), ...
%          'HorizontalAlignment', 'center', 'FontSize', 8, ...
%          'Color', [0.4 0.4 0.4]);
% end

% --- Individual datapoints overlaid on Data bars ---
% We need to map conditions to bar positions
% Bar positions follow the order of cond_names which has 8 conditions

cond_to_pts = {OR_pts, RO_pts, ORO10_pts, ORO5_pts, ORO515_pts, ROR10_pts, ROR5_pts, ROR515_pts};

for k = 1:8
    if isempty(cond_to_pts{k})
        continue
    end
    pts = cond_to_pts{k};
    n   = numel(pts);

    % Jitter x positions slightly for visibility
    jitter_width = 0.05;
    x_jitter = x_data(k) + jitter_width * randn(n, 1);

    scatter(x_jitter, pts, 30, ...
            'MarkerFaceColor', [0.2 0.4 0.8], ...
            'MarkerEdgeColor', 'w', ...
            'MarkerFaceAlpha', 0.9, ...
            'LineWidth', 0.5, ...
            'HandleVisibility', 'off');
end

% --- SEM error bars on Data bars (b(1)) ---
errorbar(x_data, targets, targets_sem, ...
         'k', 'LineStyle', 'none', 'LineWidth', 1.2, ...
         'CapSize', 5, 'HandleVisibility', 'off');

% Vertical separators between groups
xline(2.5, 'k:', 'LineWidth', 1, 'HandleVisibility', 'off');
xline(5.5, 'k:', 'LineWidth', 1, 'HandleVisibility', 'off');

xticks(1:8);
xticklabels(cond_names);
xtickangle(30);
ylabel('Weight change [%]');
ylim([-40 60]);   % expanded slightly to accommodate individual points
legend({'Data', 'Model'}, 'Location', 'northwest');
grid on; box off;

% Style
ax = gca;
ax.Color           = [0.93 0.93 0.93];
ax.GridColor       = [1 1 1];
ax.GridAlpha       = 1.0;
ax.GridLineStyle   = '-';
ax.XGrid           = 'on';
ax.YGrid           = 'on';
ax.MinorGridLineStyle = 'none';
ax.Box             = 'off';
ax.XAxis.TickDirection = 'out';
ax.YAxis.TickDirection = 'out';
ax.FontName        = 'Arial';
ax.FontSize        = 13;
set(gcf, 'Color', 'w');

% ---- Figure 2: Scatter plot ----
figure('Position', [100 100 500 500]);
hold on;

% Unity line
all_vals = [targets, pred];
ax_min   = min(all_vals) - 5;
ax_max   = max(all_vals) + 5;
plot([ax_min ax_max], [ax_min ax_max], 'k--', 'LineWidth', 1.2, ...
     'HandleVisibility', 'off');

% Plot each condition with its group colour and label
for k = 1:8
    scatter(targets(k), pred(k), 100, colors(k,:), ...
            'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.8, ...
            'HandleVisibility', 'off');

    % Offset labels to avoid overlap with points
    x_off = 0.8;
    y_off = 0.8;
    text(targets(k) + x_off, pred(k) + y_off, ...
         cond_names{k}, 'FontSize', 8, 'Color', colors(k,:));
end

% Dummy points for legend
scatter(nan, nan, 80, [0.2 0.2 0.8], 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Single Pair');
scatter(nan, nan, 80, [0.8 0.2 0.2], 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Post-Pre-Post');
scatter(nan, nan, 80, [0.1 0.6 0.1], 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Pre-Post-Pre');

% R² annotation
ss_res = sum((pred - targets).^2);
ss_tot = sum((targets   - mean(targets)).^2);
r2     = 1 - ss_res/ss_tot;
text(ax_min + 1, ax_max - 2, sprintf('R^2 = %.2f', r2), ...
     'FontSize', 11, 'FontWeight', 'bold');

xlim([ax_min ax_max]); ylim([ax_min ax_max]);
axis square;
xlabel('Target \Deltaw [%]');
ylabel('Predicted \Deltaw [%]');
title('Predictive Dendrites Model: Goodness of Fit');
legend('Location', 'southeast');
grid on; box off;

% --------------------------------------------------------
%% Objective function
function [sse, predictions] = compute_residuals(p, targets, dt, T, t_anchor, g)

lambda     = p(1);
epsilon    = p(2);
w0         = p(3);
tau        = p(4);
alpha      = p(5);
delay_samp = round(tau / dt);

% Define the 8 conditions
%
% Convention: post - pre = dT
%
% OR (-10ms): pre at t_anchor, post at t_anchor - 10
%         (post precedes pre by 10ms => post-pre = -10)
% RO (+10ms): pre at t_anchor, post at t_anchor + 10
%         (post follows pre by 10ms => post-pre = +10)
% ORO-10: post-pre-post {-10,+10}: one pre at t_anchor,
%         two posts at t_anchor-10 (dT=-10) and t_anchor+10 (dT=+10)
% ORO-5: post-pre-post {-5,+5}: one pre at t_anchor,
%         two posts at t_anchor-5 (dT=-5) and t_anchor+5 (dT=+5)
% ORO(-5,+15): post-pre-post {-5,+15}: one pre at t_anchor,
%         two posts at t_anchor-5 (dT=-5) and t_anchor+15 (dT=+15)
% ROR-10: pre-post-pre {+10,-10}: one post at t_anchor,
%         two pres at t_anchor-10 (dT=+10) and t_anchor+10 (dT=-10)
% ROR-5: pre-post-pre {+5,-5}: one post at t_anchor,
%         two pres at t_anchor-5 (dT=+5) and t_anchor+5 (dT=-5)
% ROR(+5,-15): pre-post-pre {+5,-15}: one post at t_anchor,
%         two pres at t_anchor-5 (dT=+5) and t_anchor+15 (dT=-15)

conditions = {
    struct('pre', t_anchor,                      'post', t_anchor-10          );  % OR  -10
    struct('pre', t_anchor,                      'post', t_anchor+10          );  % RO  +10
    struct('pre', t_anchor,                      'post', [t_anchor-10, t_anchor+10]);  % ORO-10
    struct('pre', t_anchor,                      'post', [t_anchor-5,  t_anchor+5 ]);  % ORO-5
    struct('pre', t_anchor,                      'post', [t_anchor-5,  t_anchor+15]);  % ORO(-5,+15)
    struct('pre', [t_anchor-10, t_anchor+10],    'post', t_anchor             );  % ROR-10
    struct('pre', [t_anchor-5,  t_anchor+5 ],    'post', t_anchor             );  % ROR-5
    struct('pre', [t_anchor-5,  t_anchor+15],    'post', t_anchor             );  % ROR(+5,-15)
};

predictions = zeros(1, 8);

for k = 1:8
    c = conditions{k};

    % Build spike trains
    s_pre  = zeros(1, T);
    s_post = zeros(1, T);

    ip = round(c.pre  / dt) + 1;
    ip = ip(ip >= 1 & ip <= T);
    s_pre(ip) = 1;

    ipo = round(c.post / dt) + 1;
    ipo = ipo(ipo >= 1 & ipo <= T);
    s_post(ipo) = 1;

    % Initialise state variables
    x_out = zeros(1, T);
    x_d   = zeros(1, T);
    v_d   = zeros(1, T);
    w     = zeros(1, T);
    w(1)  = w0;

    for t = 2:T
        % Update activity traces
        x_d(t)   = x_d(t-1)   + dt * (-lambda * x_d(t-1))   + s_pre(t);
        x_out(t) = x_out(t-1) + dt * (-lambda * x_out(t-1)) + s_post(t);

        % Delayed index — clamp to 1 to avoid zero index
        t_del = max(t - delay_samp, 1);

        % Dendritic potential uses delayed presynaptic activity
        v_d(t) = w(t-1) * x_d(t_del);

        % Weight update
        dw   = alpha * (x_out(t) - v_d(t)) * x_d(t_del) * g(x_out(t), epsilon);
        w(t) = w(t-1) + dt * dw;
    end

    predictions(k) = (w(end) - w(1)) / w(1) * 100;
end

sse         = sum((predictions - targets).^2);
end