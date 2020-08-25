



%% Colors 
SveAllpics=1;
closefigs=0;
prefix='D:\Neuromatch\Project\Analysis\Plots\';
pathsave=prefix;
  existance=exist(strcat(pathsave));
        if existance==0
            [s,m,mm]=mkdir(pathsave);
            prefix = strcat(pathsave);
        else
            delete([pathsave '*.fig'])
            delete([pathsave '*.png'])
            delete([pathsave '*.pdf'])
            delete([pathsave '*.eps'])
            %             rmdir([pathsave,'FigOutput'],'s'); %To erase the folder
            prefix = strcat(pathsave);
        end








%%
Gen_brain_a={'Vis','Motor','Thal','Hippo'};
Orange=[247 148 29]./255; %Right
Green=[0 166 81]./255;%Left
Grey=[188 190 192]/255;% No go
Wrong = [0 0 0];% Wrong

VIS= [157 173 214]./255;
MO=[162 142 120]./255;
Thal= [193 219 172]./255;
Hippo=[196 155 182]./255;

clrs=[VIS;MO;Thal;Hippo];
clrs2=[Orange;Green;Grey;Wrong];
figure; 
for ii=1:4
    datin=movmean(squeeze(mean(mean(alldat.spks(Brain_areas_match_spks{ii},resp_R_cort,:))))-...
         mean(squeeze(mean(mean(alldat.spks(Brain_areas_match_spks{ii},resp_R_cort,1:50))))),3).*10;
     
    line([0, 0],[-0.2 65],'color',[0.8 0.8 0.8],'linestyle',':')
    hold on
    plot(trial_t, datin,'Color',clrs(ii,:),'linewidth',2)
    hold on
    plot(trial_t(58),datin(58) ,'*','color',[1 1 1].*0.35,'linewidth',5)
        hold on
    plot(trial_t(77),datin(77) ,'*','color',[1 1 1].*0.35,'linewidth',5)
    xlabel('Time(ms)')
    ylabel('Amplitude')
    title('Spikes')
    set(gca, 'FontSize',14)
end 
ylim([-0.1 0.45])
fname =['Slide_spks_Example'];
hh=gcf;
set(hh,'name',fname,'Position',[ 510   723   560   221])

         if SveAllpics==1
               h = gcf;
               saveas(h,strcat(prefix,fname,'.fig'),'fig');
               saveas(h,strcat(prefix,fname,'.png'),'png');
               print(h,'-dpng', [prefix,fname], '-r600');
               print(h,'-dpdf', [prefix,fname], '-r600');
%                print(h,'-depsc', [prefix,fname], '-r600');
           end
         if closefigs==1, close all,  end 






figure;
for ii=1:4
    
    datin=movmean(squeeze(mean(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_R_cort,:))))-...
        mean(squeeze(mean(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_R_cort,1:50))))),8).*10;
    line([0, 0],[-80 35],'color',[0.8 0.8 0.8],'linestyle',':')
    hold on
    plot(trial_t,datin','color',clrs(ii,:),'linewidth',2)
    hold on
    plot(trial_t(58),datin(58) ,'*','color',[1 1 1].*0.35,'linewidth',5)
        hold on
    plot(trial_t(77),datin(77) ,'*','color',[1 1 1].*0.35,'linewidth',5)
    hold on

    xlabel('Time(ms)')
    ylabel('Amplitude')
    title('LFP')
    set(gca, 'FontSize',14)
        
end 
ylim([-80 40])

fname =['Slide_LFP_Example'];
hh=gcf;
set(hh,'name',fname,'Position',[ 510   723   560   221])

         if SveAllpics==1
               h = gcf;
               saveas(h,strcat(prefix,fname,'.fig'),'fig');
               saveas(h,strcat(prefix,fname,'.png'),'png');
               print(h,'-dpng', [prefix,fname], '-r600');
               print(h,'-dpdf', [prefix,fname], '-r600');
%                print(h,'-depsc', [prefix,fname], '-r600');
           end
         if closefigs==1, close all,  end 

    
         
         VIS= [157 173 214]./255;
         MO=[162 142 120]./255;
         Thal= [193 219 172]./255;
        Hippo=[196 155 182]./255;
         clrs=[VIS;MO;Thal;Hippo];
         
         figure;
         b =bar([85 70 65 61]./100)
         b.FaceColor = 'flat';
         b.EdgeColor = 'flat'
         for ii=1:4
         b.CData(ii,:) = clrs(ii,:);
         end 
         set(gca,'XTickLabel',Gen_brain_a)
         ylabel('Accuracy')
         set(gca, 'FontSize',14)
         ylim([0 1])
         box off
         fname =['Slide_Hypothetical_Pred'];
         hh=gcf;
         set(hh,'name',fname,'Position',[680   854   360   244])

         if SveAllpics==1
               h = gcf;
               saveas(h,strcat(prefix,fname,'.fig'),'fig');
               saveas(h,strcat(prefix,fname,'.png'),'png');
               print(h,'-dpng', [prefix,fname], '-r600');
               print(h,'-dpdf', [prefix,fname], '-r600');
%                print(h,'-depsc', [prefix,fname], '-r600');
           end
         if closefigs==1, close all,  end 
         
%% Model Results


% Model_results=[  14.395000000000000  22.611000000000001   6.582000000000000  11.270000000000000  22.893000000000001  34.988000000000000;...
%    9.851000000000001  21.728000000000002   0.720000000000000   3.733000000000000  23.856999999999999  29.683000000000000;...
%    9.532000000000000  21.077999999999999   1.028000000000000   3.480000000000000  23.263000000000002  29.292000000000002;...
%   18.042999999999999  26.739000000000001   6.054000000000000  10.808999999999999  26.876000000000001  39.857999999999997;...
%   16.463823999999999  29.209080000000000  31.037156000000000  38.016164000000003  39.655000000000001  46.876283999999998]


Model_results=[  14.395000000000000   6.085000000000000  22.611000000000001   6.582000000000000  11.270000000000000  22.893000000000001  34.988000000000000;...
   9.851000000000001   4.185000000000000  21.728000000000002   0.720000000000000   3.733000000000000  23.856999999999999  29.683000000000000;...
   9.532000000000000   3.065000000000000  21.077999999999999   1.028000000000000   3.480000000000000  23.263000000000002  29.292000000000002;...
  18.042999999999999   6.662000000000000  26.739000000000001   6.054000000000000  10.808999999999999  26.876000000000001  39.857999999999997;...
  16.463823999999999  35.574880000000000  29.209080000000000  31.037156000000000  38.016164000000003  39.655000000000001  46.876283999999998]


Mean_Acc=[46.3950000000000,37.8590000000000,54.6330000000000,38.6580000000000,43.3320000000000,54.6330000000000,62.5000000000000;...
               36.3300000000000,30.6770000000000,48.1100000000000,27.2500000000000,30.2030000000000,50.3590000000000,55.7330000000000;...
               35.9590000000000,29.7650000000000,47.7890000000000,27.5300000000000,30.0380000000000,50.0350000000000,55.7710000000000;...
               48.3550000000000,36.7790000000000,56.4070000000000,35.9930000000000,40.6230000000000,57.2610000000000,67;...
               79.1530000000000,52.3160000000000,63.4980000000000,47.6030000000000,58.3070000000000,79.3100000000000,90.1467000000000];
           
Mean_Acc_Chance=[32,31.7740000000000,32.0220000000000,32.0760000000000,32.0620000000000,31.7400000000000,27.5120000000000;...
    26.4790000000000,26.4920000000000,26.3820000000000,26.5300000000000,26.4700000000000,26.5020000000000,26.0500000000000;...
    26.4270000000000,26.7000000000000,26.7110000000000,26.5020000000000,26.5580000000000,26.7720000000000,26.4790000000000;...
    30.3120000000000,30.1170000000000,29.6680000000000,29.9390000000000,29.8140000000000,30.3850000000000,27.1420000000000;...
    0.208000000000000,0.680000000000000,0.460000000000000,0.652000000000000,0.652000000000000,0.500000000000000,0.520000000000000];         

Models_used={'SVM','Decision Tree','AdaBoost','Random Forest','Neural Net'};           
Area_to_plot=2:5;
for nn=1:length(Models_used) %B/C we used 5 models
    if nn==5,
    chnc=.25;    
    else
    chnc=mean(Mean_Acc_Chance(nn,Area_to_plot))./100;
    end 
         figure;
         b =bar(Mean_Acc(nn,Area_to_plot)./100)
         b.FaceColor = 'flat';
         b.EdgeColor = 'flat'
         for ii=1:4
         b.CData(ii,:) = clrs(ii,:);
         end 
         set(gca,'XTickLabel',Gen_brain_a)
         hold on
         line([0, 6],[1 1].*chnc,'color',[1 1 1].*0.35,'linestyle',':')
         ylabel('Accuracy')
         title([Models_used{nn}])
         set(gca, 'FontSize',14)
%          ylim([0 40])
         box off
         

ylim([0 .70])
xlim([0 5])

         fname =['Slide_Model_Pred_',Models_used{nn}];
         hh=gcf;
         set(hh,'name',fname,'Position',[680   854   360   244])
         if SveAllpics==1
               h = gcf;
               saveas(h,strcat(prefix,fname,'.fig'),'fig');
               saveas(h,strcat(prefix,fname,'.png'),'png');
               print(h,'-dpng', [prefix,fname], '-r600');
               print(h,'-dpdf', [prefix,fname], '-r600');
           end
         if closefigs==1, close all,  end 

end 
         

mouse_clrs=[[69, 123, 157]./255;[131, 197, 190]./255]

Models_used={'SVM','Decision Tree','AdaBoost','Random Forest','Neural Net'};           
Area_to_plot=6:7;
for nn=1:length(Models_used) %B/C we used 5 models
    if nn==5,
    chnc=.25;    
    else
    chnc=mean(Mean_Acc_Chance(nn,Area_to_plot))./100;
    end 
         figure;
         b =bar(Mean_Acc(nn,Area_to_plot)./100)
         b.FaceColor = 'flat';
         b.EdgeColor = 'flat';
         for ii=1:2
         b.CData(ii,:) =  mouse_clrs(ii,:);
         end 
         set(gca,'XTickLabel',{['Several'], ['Single']})
         xlabel(['Mice'])
         hold on
         line([0, 3],[1 1].*chnc,'color',[1 1 1].*0.35,'linestyle',':')
         ylabel('Accuracy')
         title([Models_used{nn}])
         set(gca, 'FontSize',14)
%          ylim([0 40])
         box off
         

ylim([0 1])
xlim([0 3])

         fname =['Slide_Model_Pred_SeveralVSingle_',Models_used{nn}];
         hh=gcf;
         set(hh,'name',fname,'Position',[ 703   457   318   271])
         if SveAllpics==1
               h = gcf;
               saveas(h,strcat(prefix,fname,'.fig'),'fig');
               saveas(h,strcat(prefix,fname,'.png'),'png');
               print(h,'-dpng', [prefix,fname], '-r600');
               print(h,'-dpdf', [prefix,fname], '-r600');
           end
         if closefigs==1, close all,  end 

end 
         



