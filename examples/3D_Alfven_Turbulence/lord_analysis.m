%%
clc
clear
addpath("D:/LoRD-main/matlab/");
%%
clc
python_exe = 'C:\Users\mrwwn\AppData\Local\Programs\Python\Python39\python.exe';  % Windows
% python_exe = '/usr/local/bin/python3.9';  % Linux/macOS

% 检查 Python 可执行文件是否存在
if ~exist(python_exe, 'file')
    error('Python 可执行文件不存在: %s', python_exe);
end

% 设置 Python 环境
pyenv('Version', python_exe);

% 验证环境是否加载成功
% p = pyenv;
% if ~strcmp(p.Status, 'Loaded')
%     error('Python 环境加载失败！状态: %s', p.Status);
% end
% 
% fprintf('Python 环境加载成功！版本: %s\n', p.Version);
% 获取当前 MATLAB 脚本所在目录（假设 Python 文件也在此目录）
current_dir = pwd;

% 将目录添加到 Python 搜索路径的首位
py.sys.path().insert(int32(0), current_dir);

% 显示 Python 搜索路径（用于调试）
fprintf('Python 搜索路径:\n');
path = py.sys.path();
for i = 1:length(py.sys.path())
    
    fprintf('  %d: %s\n', i, char(path{i}));
end
%%
clc
% 添加Python脚本所在目录到MATLAB路径
%pyenv('Version', 'C:\Users\mrwwn\AppData\Local\Programs\Python\Python39\python.exe');
%addpath(genpath('D:\Research\Codes\Hybrid-vpic\'));
py.sys.path().insert(int32(0), 'D:\Research\Codes\Hybrid-vpic\');
% 调用Python函数读取.gda文件
%file_path = 'your_file.gda';
field_dir = "D:\Research\Codes\Hybrid-vpic\HybridVPIC-main\3Dalfven-turbulence\data_0resistivity\";
% if py.help('read_field_data.loadinfo') ~= py.none
%     fprintf('函数存在！\n');
% else
%     fprintf('函数不存在！\n');
% end
nxyz_data = int32(py.read_field_data.loadinfo(field_dir));
nx = nxyz_data(1);
ny = nxyz_data(2);
nz = nxyz_data(3);
epoch = 4;
bx_data = double(py.read_field_data.load_data_at_certain_t(strcat(field_dir,"bx.gda"), int32(epoch), nx, ny, nz));
by_data = double(py.read_field_data.load_data_at_certain_t(strcat(field_dir,"by.gda"), int32(epoch), nx, ny, nz));
bz_data = double(py.read_field_data.load_data_at_certain_t(strcat(field_dir,"bz.gda"), int32(epoch), nx, ny, nz));
ex_data = double(py.read_field_data.load_data_at_certain_t(strcat(field_dir,"ex.gda"), int32(epoch), nx, ny, nz));
ey_data = double(py.read_field_data.load_data_at_certain_t(strcat(field_dir,"ey.gda"), int32(epoch), nx, ny, nz));
ez_data = double(py.read_field_data.load_data_at_certain_t(strcat(field_dir,"ez.gda"), int32(epoch), nx, ny, nz));
pxx_data = double(py.read_field_data.load_data_at_certain_t(strcat(field_dir,"pi-xx.gda"), int32(epoch), nx, ny, nz));
pyy_data = double(py.read_field_data.load_data_at_certain_t(strcat(field_dir,"pi-yy.gda"), int32(epoch), nx, ny, nz));
pzz_data = double(py.read_field_data.load_data_at_certain_t(strcat(field_dir,"pi-zz.gda"), int32(epoch), nx, ny, nz));
ni_data = double(py.read_field_data.load_data_at_certain_t(strcat(field_dir,"ni.gda"), int32(epoch), nx, ny, nz));
T_data = (pzz_data+pxx_data+pyy_data)/3./ni_data;

%% 步骤1：生成均匀网格和示例磁场数据（11x11x11网格）
clc
x = linspace(1, 256, 256);  % x方向坐标 [-1,1]
y = linspace(1, 256, 256);  % y方向坐标 [-1,1]
z = linspace(1, 256, 256);   % z方向坐标 [0,2]
[X, Y, Z] = meshgrid(x, y, z);
%%
disp(min(bx_data,[],"all"))
%%
isovalues = [0.1, 0.2, 0.3];
colors = jet(length(isovalues));  % 为每个等值面分配颜色

figure;
hold on;

% 循环绘制每个等值面
for i = 1:length(isovalues)
    disp(i)
    [F, V] = isosurface(x, y, z, bx_data, isovalues(i));
    patch('Faces', F, 'Vertices', V, 'FaceColor', colors(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.1);
end

% 设置视角、光照和标签
view(3);
axis equal tight;
camlight;
lighting gouraud;
colorbar;
%%
% % 创建物理合理的磁场分布（满足∇·B≈0）
% Bx =  Z .* exp(-X.^2 - Y.^2);       % x分量：高斯分布随z增大
% By = -X .* sin(pi*Z);               % y分量：剪切场
% Bz = 0.5 * Y .* cos(2*pi*X) + 0.3;  % z分量：振荡场+背景场
% 
% % 添加局部电流片特征（重联位点）
% current_sheet = abs(Y) < 0.2 & abs(Z-1) < 0.3;
% Bx(current_sheet) = Bx(current_sheet) + 5 * tanh(X(current_sheet)*10);
% By(current_sheet) = By(current_sheet) - 3 * sech(Y(current_sheet)*8);
Bx = bx_data;
By = by_data;
Bz = bz_data;
%% 步骤2：配置ARD参数
Parameters = struct();
Parameters.ARD_AnalyzeLocalEffects=1;
Parameters.ARD_AnalyzeAllGrids = 0;       % 启用阈值筛选
Parameters.ARD_ScalarThreshold = 0.02;     % |E∥|阈值
Parameters.ARD_ShowThresScalarProfile = 0; % 显示阈值直方图
Parameters.OutputType = 'mat';             % 输出.mat文件
Parameters.OutputLabel = 'demo';           % 输出文件名标签
Parameters.NumRAMBlock = 2;                % 内存分块处理
Parameters.ARD_FixTrace = 0;
Parameters.OutputDir="D:\Research\Codes\Hybrid-vpic\HybridVPIC-main\3Dalfven-turbulence\";
% 使用默认焦耳耗散模型（ηJ），故不提供N1,N2,N3

%% 步骤3：调用ARD函数分析重联位点
RDInfo = ARD(Bx, By, Bz, x, y, z, Parameters, ex_data, ey_data, ez_data);
%%
clc
disp(max(ez_data, [], "all"))

%% 步骤4：可视化结果
% 提取分析结果
RDGrids = RDInfo.Data(:, 1:3);       % 网格坐标
Types = RDInfo.Data(:, 4);           % 磁场类型
IsExtreme = RDInfo.Data(:, 5) == 1;  % 极值点标记
Angles = RDInfo.Data(:, 6);          % 特征角θ_eig
[RMarker, RGridCount]=Tool_Cluster_RegionGrow(RDInfo.Data(:, 1),RDInfo.Data(:, 2),RDInfo.Data(:, 3),x,y,z);
%%
x_RD = Tool_ARD_ReadRDInfo(RDInfo,'x1');
y_RD = Tool_ARD_ReadRDInfo(RDInfo,'x2');
z_RD = Tool_ARD_ReadRDInfo(RDInfo,'x3');
RDType = Tool_ARD_ReadRDInfo(RDInfo,'RDType');
Is2DExtrema = logical(Tool_ARD_ReadRDInfo(RDInfo,'Is2DExtrema'));
B0_RD = Tool_ARD_ReadRDInfo(RDInfo,'B0','Extra'); % Read from RDInfo.ExtraData
[X1,X2,X3] = meshgrid(x,y,z);

%% Draw figure
figure1 = figure('Units','centimeters','Position',[1 1 15 8.5]);
axes1 = axes('Parent',figure1,'Position',[0.1,0.07,0.85,0.95]);
hold(axes1,'on');
box(axes1,'on');
grid(axes1,'on');
hal = xlabel('$x_1\,\left(\mathrm{Mm}\right)$','Interpreter','latex');
set(hal,'Position',[0,-250,0])
hal = ylabel('$x_2\,\left(\mathrm{Mm}\right)$','Interpreter','latex');
set(hal,'Position',[-270,0,0])
hal = zlabel('$x_3\,\left(\mathrm{Mm}\right)$','Interpreter','latex');
set(hal,'Position',[-240,180,100])

b_RD = RDType == 1 & Is2DExtrema; % X-type: 2D extremal Epara
plot3(x_RD(b_RD),y_RD(b_RD),z_RD(b_RD),'.','MarkerSize',2);
b_RD = RDType == 2 | RDType == 3 & Is2DExtrema; % O-type: 2D extremal Epara
plot3(x_RD(b_RD),y_RD(b_RD),z_RD(b_RD),'.','MarkerSize',1);

hs = slice(X1,X2,X3,Bz,[],[],[0]); 
set(hs,'EdgeColor','none');
colormap('gray');

set(axes1,'XLim',[-200,200],'YLim',[-180,180],'ZLim',[0,200],...
    'DataAspectRatio',[1,1,1],...
    'TickDir','in','layer','top','TickLabelInterpreter','latex',...
    'FontSize',8,'Projection','perspective', 'BoxStyle','back');
view(axes1,[-35 12]);
%%
clc
R0_O = [-41.47 4.29 131.56]; % Position of sphere center
Radius_O = 20; % Radius of sphere
n_sample_O = 20; % Number of initial samplings
LineConf.Len = 10000;
LineConf.Color = 'k'; % Field-line color
LineConf.Style = '-'; % Field-line style
LineConf.Width = 0.5; % Field-line width
hflO = Tool_PreviewFieldLines(Bx,By,Bz,x,y,z,'l',...
                              R0_O,Radius_O,n_sample_O,LineConf);
%%
histogram(RGridCount(:,2))
set(gca,"YScale",'log')
%%
clc
clear B N
N.e1=ex_data;N.e2=ey_data;N.e3=ez_data;
B.e1=Bx;B.e2=By;B.e3=Bz;
length(fieldnames(B))
e_parallel = func_m_VectorDot(N,func_m_VectorDirection(B));
disp(max(e_parallel,[],'all'))
%%
clc
grid_of_class = RDGrids(class_idx,:);
%test = grid_of_class(4,:);
test=sub2ind([256, 256, 256],grid_of_class(4,1),grid_of_class(4,2),grid_of_class(4,3));
disp(length(e_parallel(abs(e_parallel)>=0.05)))
condition = find(abs(e_parallel)>=0.05);
[a,b,c]=ind2sub([256,256,256],condition);
disp(abs(e_parallel(104,16,1)))
%%
clear G
G.e1=X;G.e2=Y;G.e3=Z;
func_m_TensorSubset(G,(abs(e_parallel)>=0.05))
%%
%disp(mean(T_data,'all'))
disp(length(T_data(T_data>0.35))/length(T_data(T_data>0)))
%%
T_for_each_class_4 = zeros(length(RGridCount),1);
x_range=[0,210];
y_range=[0,180];
z_range=[0,180];
for i=1:length(RGridCount)
    disp(i)
class_idx = RMarker==RGridCount(i,1);
grid_of_class = RDGrids(class_idx,:);
if min(grid_of_class(:,2))>x_range(2) || max(grid_of_class(:,2))<x_range(1) || min(grid_of_class(:,1))>y_range(2) || max(grid_of_class(:,1))<y_range(1) || min(grid_of_class(:,3))>z_range(2) || max(grid_of_class(:,3))<z_range(1)
    continue
end
linear_indices = sub2ind([256, 256, 256], int32(grid_of_class(:,2)), int32(grid_of_class(:,1)), int32(grid_of_class(:,3)));
e_parallel_mask=e_parallel(linear_indices);
T_mask = T_data(linear_indices);
T_for_each_class_4(i) = mean(T_mask);
scatter3(grid_of_class(:,2), grid_of_class(:,1), grid_of_class(:,3),[],T_mask,'filled',SizeData=5,MarkerFaceAlpha=0.7)
n = 256;  % 颜色数量
% bwr_map = [linspace(0, 1, n)', linspace(0, 0, n)', linspace(1, 0, n)'];
colormap(jet)
hold on
end
clim([0.3,0.35])
cbar=colorbar();
ylabel(cbar,"E_{parallel}")
ylabel(cbar,"Temperature")
% xlim([1,256])
% ylim([1,256])
% zlim([1,256])
xlim([0,210])
ylim([0,180])
zlim([0,180])
xlabel("x")
ylabel("y")
zlabel("z")
title("t=136\omega^{-1}_{ci},E_{threshold}=0.03",fontsize=15)
box on
%%
h1=histogram(T_for_each_class_1,linspace(0.25, 0.4, 20),'FaceAlpha',0.5,'normalization','probability','FaceColor','w','EdgeColor','b','LineWidth',1.5);
%legend("t=68\omega_{ci}^{-1}")
hold on

set(gca,'YScale','log')
h2=histogram(T_for_each_class_4,linspace(0.25, 0.4, 20),'FaceAlpha',0.5,'normalization','probability','FaceColor','w','EdgeColor','r','LineWidth',1.5);
%legend([h1,h2],{"t=68\omega_{ci}^{-1}","t=272\omega_{ci}^{-1}"})
hold on
h3=histogram(T_for_each_class_2,linspace(0.25, 0.4, 20),'FaceAlpha',0.5,'normalization','probability','FaceColor','w','EdgeColor','g','LineWidth',1.5);
%legend("t=68\omega_{ci}^{-1}")
hold on
legend([h1,h3,h2],{"t=68\omega_{ci}^{-1}","t=136\omega_{ci}^{-1}","t=272\omega_{ci}^{-1}"})
set(gca,'YScale','log')
xlabel("temperature", FontSize=15)
ylabel("probability", FontSize=15)
%%
e_parallel = (ex_data.*bx_data+ey_data.*by_data+ez_data.*bz_data)./sqrt(bx_data.^2+by_data.^2+bz_data.^2);
for i=1:10
class_idx = RMarker==RGridCount(i,1);
grid_of_class = RDGrids(class_idx,:);
linear_indices = sub2ind([256, 256, 256], grid_of_class(:,2), grid_of_class(:,1), grid_of_class(:,3));
e_parallel_mask=e_parallel(linear_indices);
scatter3(grid_of_class(:,2), grid_of_class(:,1), grid_of_class(:,3),[],Types(class_idx),'filled',SizeData=20,AlphaData=0.5)
n = 256;  % 颜色数量
% bwr_map = [linspace(0, 1, n)', linspace(0, 0, n)', linspace(1, 0, n)'];
% colormap(bwr_map)
hold on
end
colormap(jet(5))
cbar=colorbar();
ylabel(cbar,"E_{parallel}")
% xlim([1,256])
% ylim([1,256])
% zlim([1,256])
xlabel("x")
ylabel("y")
zlabel("z")
%%
histogram(Types)
xticks(1:5);  % 设置刻度位置为1-9
xticklabels({
    '3D X', 
    '3D O(Repelling)', 
    '3D O(Attracting)', 
    '3D Repelling', 
    '3D Attracting'
});  % 标签顺序对应表1中的类型定义
ylabel("Counts", FontSize=15)
%%
% 选择目标类型（例如Type 1：3D X型，可根据需要修改为其他类型）
targetType = 1;
fig = figure(1);
for i=1:5
typeLabel = '3D X';  % 对应表1中的类型名称{insert\_element\_0\_}

% 筛选该类型的特征角数据
typeIdx = Types == i;
anglesOfType = Angles(typeIdx);

% 定义角度区间（按10度分组，覆盖0-90度）
angleBins = 0:30:90;

% 统计每个区间的数量
[counts, edges] = histcounts(anglesOfType, angleBins);

% 创建堆叠直方图（单一类型下按角度区间堆叠，此处用不同颜色区分区间）
%figure('Position',[200 200 800 600]);
h = bar(i, counts, 'stacked', 'FaceColor', 'flat','HandleVisibility','off');
h(3).CData=[1,0,0];
h(2).CData=[0,1,0];
h(1).CData=[0,0,1];
if i==1
    legend(h,{'0°<\theta_{eig}<30°','30°<\theta_{eig}<60°','60°<\theta_{eig}<90°'})
end
hold on
end
% 设置颜色映射（区分不同角度区间）
%colormap(jet(length(edges)-1));

% 添加标签与标题
%xlabel('\theta_{eig} (degrees)', 'FontSize', 12);
xticks(1:5);
xticklabels({
    '3D X', 
    '3D O(Repelling)', 
    '3D O(Attracting)', 
    '3D Repelling', 
    '3D Attracting'
});  % 标签顺序对应表1中的类型定义
ylabel('Counts', 'FontSize', 12);
title(['Type ', num2str(targetType), ' (', typeLabel, ') 的\theta_{eig}分布（堆叠形式）'], 'FontSize', 14);
%xticks(angleBins);
%xlim([0 90]);  % 固定角度范围为0-90度{insert\_element\_1\_}
%grid on;
exportgraphics(gcf,"type_hist.png")
% 添加颜色条说明角度区间
%colorbar;
%caxis([min(edges) max(edges)]);

%ylabel(colorbar, '角度区间 (degrees)', 'FontSize', 10);
%%
% 创建3D可视化
figure('Position', [100, 100, 1200, 500])
subplot(1,2,1)

% 绘制所有识别到的重联网格
scatter3(RDGrids(:,1), RDGrids(:,2), RDGrids(:,3), 20, Types, 'filled')
title('磁场结构类型分布 (RDType)')
colormap(jet(9))  % 9种类型对应9种颜色
colorbar('Ticks',1:9, 'TickLabels',{'3D-X','3D-O(斥)','3D-O(吸)','3D-源','3D-汇','3D-反平行','2D-X','2D-O','2D-反平行'})
xlabel('X'); ylabel('Y'); zlabel('Z'); view(30,30)
%%
% 标记极值点（重联核心位点）
subplot(1,2,2)
extremeGrids = RDGrids(IsExtreme, :);
extremeAngles = Angles(IsExtreme);
scatter3(extremeGrids(:,1), extremeGrids(:,2), extremeGrids(:,3), 40, extremeAngles, 'filled')
title('重联极值点 (\nabla_\perp E_\parallel \approx 0)','Interpreter','latex')
colormap(hot); colorbar; clim([0 90])
xlabel('X'); ylabel('Y'); zlabel('Z'); view(30,30)
sgtitle('ARD重联分析结果')
