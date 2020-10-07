clear

LF= readNPY('LF_all.npy');
RF= readNPY('RF_all.npy');
LT= readNPY('LT_all.npy');
RT= readNPY('RT_all.npy');
LF = flip(LF, 2);
LT = flip(LT, 2);

FF = [RF; LF];
TT = [RT; LT];

%% Left
Lids = readNPY('Lids.npy');

%% Right
Rids = readNPY('Rids.npy');

%% JSN
JSL = readNPY('JSL.npy');
JSM = readNPY('JSM.npy');

%%
for i = 1:100
    b(2*i - 1) = i;
    b(2*i) = i;
end

%%
condition=((JSL==0) & (JSM>0));

setFigure(1000,200)
subplot(1,2,1)
set(0,'DefaultAxesFontSize',16)
boxplot(FF(condition,1:1:200),b.','Symbol','k.','BoxStyle','outline','OutlierSize',2,'Colors',[66 135 245]/255);
%ylim([0 0.025])
ylim([0 350])
xticks([1 50 100])
xticklabels({'0.0','0.5','1.0'})
title('Lateral <------|------> Medial')
xlabel('Normalized Slice Number')
ylabel('Normalized Femur Subchondral Length')

subplot(1,2,2)
set(0,'DefaultAxesFontSize',16)
boxplot(TT(condition,1:1:200),b.','Symbol','k.','BoxStyle','outline','OutlierSize',2,'Colors',[66 135 245]/255);
%ylim([0 0.015])
ylim([0 250])
xticks([1 50 100])
xticklabels({'0.0','0.5','1.0'})
title('Lateral <------|------> Medial')
xlabel('Normalized Slice Number')
ylabel('Normalized Tibia Subchondral Length')


%% f test
var0  = (sum(FF(condition, :), 2));
var1  = (sum(TT(condition, :), 2));
[h, p] = vartest2(var0, var1,'Tail','right');