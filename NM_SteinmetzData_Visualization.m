

clear all
close all


%%
SveAllpics=1; 
closefigs =1; 
pull_out_feat=0;

 pathsave=strcat('D:\Neuromatch\Project\Analysis\Results\');

    prefix = strcat(pathsave);

    if SveAllpics==1 %1-Save output pics, 0-Don'd save output pics
        
        existance=exist(strcat(pathsave,'FigOutput'));
        if existance==0
            [s,m,mm]=mkdir(pathsave,'FigOutput');
            prefix = strcat(pathsave,'FigOutput','\');
        else
            delete([pathsave 'FigOutput\*.fig'])
            delete([pathsave 'FigOutput\*.png'])
            delete([pathsave 'FigOutput\*.pdf'])
            delete([pathsave 'FigOutput\*.eps'])
            %             rmdir([pathsave,'FigOutput'],'s'); %To erase the folder
            prefix = strcat(pathsave,'FigOutput','\');
        end
    end 


%%






DataSets_lfp={'dat_lfp_11.mat','dat_lfp_12.mat','dat_lfp_7.mat','dat_lfp_9.mat','dat_lfp_2.mat'};
DataSets=   {'dat11.mat','dat12.mat','dat7.mat','dat9.mat','dat2.mat'};

clear AllData
for ii=1:length(DataSets)
% ii=1

aa=load(['D:\Neuromatch\Project\Data\DatMats\', DataSets_lfp{ii}]);
bb=load(['D:\Neuromatch\Project\Data\DatMats\', DataSets{ii}]);

% Merge Data
names = [fieldnames(aa); fieldnames(bb)];

AllData{ii,:} = cell2struct([struct2cell(aa); struct2cell(bb)], names, 1);

for jj=1:length(AllData{ii,:}.brain_area_lfp)
    charr=strtrim(AllData{ii,:}.brain_area_lfp(jj,:));
%     alldat.brain_area_lfp(ii,:)=[]
    AllData{ii,:}.brain_area_lfp_sort{jj,:}=charr;
end 

for jj=1:length(AllData{ii,:}.brain_area)
    charr=strtrim(AllData{ii,:}.brain_area(jj,:));
%     alldat.brain_area_lfp(ii,:)=[]
    AllData{ii,:}.brain_area_spks_sort{jj,:}=charr;
end 


end


% alldat = cell2struct([struct2cell(aa); struct2cell(bb)], names, 1);
% for jj=1:length(alldat.brain_area)
%     charr=strtrim(alldat.brain_area(jj,:));
% %     alldat.brain_area_lfp(ii,:)=[]
%     alldat.brain_area_spks_sort{jj,:}=charr
% end 
clear All_feat_cat

for select_mouse=1:size(AllData,1)

alldat = AllData{select_mouse,1};



%%
Brain_group_global={'Vis','Thal','Hippo','Motor'};
brain_groups = {["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"]; % visual cortex
                ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"]; % thalamus
                ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"]; % hippocampal
                ["MD","MG","MOp","MOs","MRN"]; %Motor areas
                ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP"," TT"]; %non-visual cortex
                ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"]; % midbrain
                ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"]; % basal ganglia 
                ["BLA", "BMA", "EP", "EPd", "MEA"]};% cortical subplate

%Match areas
clear Brain_areas_match_lfp
for area=1:size(brain_groups,1)
Brain_areas_match_lfp{area,:}=ismember(alldat.brain_area_lfp_sort,brain_groups{area,:});
Brain_areas_match_spks{area,:}=ismember(alldat.brain_area_spks_sort,brain_groups{area,:});
end 

 %Plot mean trials 
 
resp_L_cort=logical((alldat.response==1).*(alldat.contrast_right<alldat.contrast_left));
resp_No_cort=logical((alldat.response==0).*(alldat.contrast_right==alldat.contrast_left));
resp_R_cort=logical((alldat.response==-1).*(alldat.contrast_right>alldat.contrast_left));
resp_Al_wrong=logical(resp_R_cort'+resp_L_cort'+resp_No_cort');




%% Plotting LFP Single Trials 


trial_t=linspace(-50,200,250);
%  figure; 
%  for ii=1:5, 
%      plot(trial_t,squeeze(mean(alldat.lfp(Brain_areas_match_lfp{ii},:,:),1))), 
%      hold on, 
%  end

 
%   figure; 
%  for ii=1:5, 
%      plot(trial_t,(1/10)*squeeze(mean(mean(alldat.spks(Brain_areas_match_spks{ii},:,:),1),2)))
%      hold on, 
%  end
%  
 
 
 %Plot one trial
 rand_trl=randperm(size(alldat.lfp,2),10);
 for tt=1:length(rand_trl)
    
  trl=rand_trl(tt);
  figure; 
  for ii=1:4; %Brain Area
  %Selected Trial
        %Magnitude of the LFP
%     datin=sqrt(movmean(squeeze(mean(alldat.lfp(Brain_areas_match_lfp{ii},trl,:),1)),5).^2);

    datin=(movmean(squeeze(mean(alldat.lfp(Brain_areas_match_lfp{ii},trl,:),1)),5));
  plot(trial_t,datin)
  hold on
  
  if alldat.response(trl)<0, answ='Right';elseif alldat.response(trl)==0 answ='None';else answ='Left'; end 
  
  if alldat.contrast_right(trl)>alldat.contrast_left(trl) && alldat.response(trl)<0
      CorNot='Correct';
  elseif alldat.contrast_right(trl)<alldat.contrast_left(trl) && alldat.response(trl)>0
      CorNot='Correct';
  elseif alldat.contrast_right(trl)==alldat.contrast_left(trl) && alldat.response(trl)==0    
      CorNot='Correct';
  else
      CorNot='Wrong';
  end 
  
  
  title({['LFP Single Trial Pooled Global Area: Trial ' num2str(trl)],...
         ['Picked:' answ,',Ans: ' CorNot ]})
  xlabel(['Time (ms)'])
  ylabel(['Voltage Amplitude'])
     
  end
  
      hold on
      
      %Plot trial start
      line_max=[60];%(max(datin)*2);
      line([0, 0],[-1,1]*line_max,'color','k','linestyle',':')
      
      
      %Plot go-cue
      gocue_t=alldat.gocue(trl)*100;
      line([1 1]*gocue_t,[-1,1]*line_max,'color','r','linestyle',':')
      
      %Plot response time
      resp_t=alldat.response_time(trl)*100;
      line([1 1]*resp_t,[-1,1]*line_max,'color','m','linestyle',':')
      
      %Plot response time
      feedbk_t=alldat.feedback_time(trl)*100;
      line([1 1]*feedbk_t,[-1,1]*line_max,'color','g','linestyle',':')
%       axis tight
legend([Brain_group_global,{'Stim Start'},{'Go'},{'Resp'},{'Rew'}])



pos = [-30 40 [1 1].*8]; 
rectangle('Position',pos,'Curvature',[1 1],'FaceColor',[1 1 1].*(1-alldat.contrast_left(trl)))
hold on 
pos = [-20 40 [1 1].*8]; 
rectangle('Position',pos,'Curvature',[1 1],'FaceColor',[1 1 1].*(1-alldat.contrast_right(trl)))
axis tight
% ylim([-60 60])


fname =['Rand_Trial_lfp', alldat.mouse_name(1:3),'_',alldat.date_exp,'_Trial_', num2str(trl)];
hh=gcf;
set(hh,'name',fname,'Position',[ 643   148   822   418])


         if SveAllpics==1
               h = gcf;
               saveas(h,strcat(prefix,fname,'.fig'),'fig');
               saveas(h,strcat(prefix,fname,'.png'),'png');
               print(h,'-dpng', [prefix,fname], '-r600');
               print(h,'-depsc', [prefix,fname], '-r600');
           end
         if closefigs==1, close all,  end 
 end 
 
 
 
 
 
 
 
%% Plotting Spikes Single Trials 
 
 
 %Plot one trial
 rand_trl=randperm(size(alldat.spks,2),10);
 for tt=1:length(rand_trl)
    
  trl=rand_trl(tt);
  figure; 
  for ii=1:4; %Brain Area
  %Selected Trial
%   ind=[4 12; ]
%   datin=movmean(squeeze(mean(alldat.lfp(Brain_areas_match_lfp{ii},trl,:),1)),5)
    datin=movmean((1/.01)*squeeze(mean(alldat.spks(Brain_areas_match_spks{ii},trl,:),1)),5);
  plot(trial_t,datin)
  hold on
  
  if alldat.response(trl)<0, answ='Right';elseif alldat.response(trl)==0 answ='None';else answ='Left'; end 
  
  if alldat.contrast_right(trl)>alldat.contrast_left(trl) && alldat.response(trl)<0
      CorNot='Correct';
  elseif alldat.contrast_right(trl)<alldat.contrast_left(trl) && alldat.response(trl)>0
      CorNot='Correct';
  elseif alldat.contrast_right(trl)==alldat.contrast_left(trl) && alldat.response(trl)==0    
      CorNot='Correct';
  else
      CorNot='Wrong';
  end 
  
  
  title({['Spks Single Trial Pooled Global Area: Trial ' num2str(trl)],...
         ['Picked:' answ,',Ans: ' CorNot ]})
  xlabel(['Time (ms)']);
  ylabel(['Spike Rate (Hz)']); 

     
  end
  
      hold on
      
      %Plot trial start
      line_max=[60];%(max(datin)*2);
      line([0, 0],[-1,1]*line_max,'color','k','linestyle',':')
      
      
      %Plot go-cue
      gocue_t=alldat.gocue(trl)*100;
      line([1 1]*gocue_t,[-1,1]*line_max,'color','r','linestyle',':')
      
      %Plot response time
      resp_t=alldat.response_time(trl)*100;
      line([1 1]*resp_t,[-1,1]*line_max,'color','m','linestyle',':')
      
      %Plot response time
      feedbk_t=alldat.feedback_time(trl)*100;
      line([1 1]*feedbk_t,[-1,1]*line_max,'color','g','linestyle',':')
%       axis tight
legend([Brain_group_global,{'Stim Start'},{'Go'},{'Resp'},{'Rew'}])



pos = [-30 20 [8 2]]; 
rectangle('Position',pos,'Curvature',[1 1],'FaceColor',[1 1 1].*(1-alldat.contrast_left(trl)))
hold on 
pos = [-20 20 [8 2]]; 
rectangle('Position',pos,'Curvature',[1 1],'FaceColor',[1 1 1].*(1-alldat.contrast_right(trl)))
axis tight
ylim([0 30])


fname =['Rand_Trial_spks', alldat.mouse_name(1:3),'_',alldat.date_exp,'_Trial_', num2str(trl)];
hh=gcf;
set(hh,'name',fname,'Position',[ 643   148   822   418])


         if SveAllpics==1
               h = gcf;
               saveas(h,strcat(prefix,fname,'.fig'),'fig');
               saveas(h,strcat(prefix,fname,'.png'),'png');
               print(h,'-dpng', [prefix,fname], '-r600');
               print(h,'-depsc', [prefix,fname], '-r600');
           end
         if closefigs==1, close all,  end 
 end 
  
 
 %% Brain Area and Stim Locs LFP
 
 
 clear  datinR  datinL  datinNo datinWr
  figure; 
  cc=1;gg=1;
  ff=1;
  for ii=1:4%:4; %Area
%   trl=109; %Selected Trial
  
  datinR=movmean(squeeze(mean(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_R_cort,:),1),2)),5);
  datinL=movmean(squeeze(mean(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_L_cort,:),1),2)),5);
  datinNo=movmean(squeeze(mean(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_No_cort,:),1),2)),5);
  datinWr=movmean(squeeze(mean(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_Al_wrong,:),1),2)),5);
  
  

      datplot1=[datinR,datinL,datinNo,datinWr];
      datplot2=[cumsum(diff(datinR).^2),cumsum(diff(datinL).^2),cumsum(diff(datinNo).^2),cumsum(diff(datinWr).^2)];


  subplot(size(datplot1,2) ,2 ,cc); cc=cc+1;
  h=plot(trial_t,datplot1,'b');
  set(h, {'color'}, num2cell(lines( size(datplot1,2) ),2));  
  xlabel(['Time(ms)'])
  ylabel(['Voltage'])
  title([Brain_group_global{ii}, '-All Correct Trials-LFP'])
  
  subplot(size(datplot1,2),2,cc); cc=cc+1;
  h=plot(trial_t(1:end-1),datplot2,'b');
  set(h, {'color'}, num2cell(lines( size(datplot2,2) ),2));  
  xlabel(['Time(ms)'])
  ylabel(['Cumulative Voltage'])
  title([Brain_group_global{ii}, '-All Correct Trials-LFP'])
    
  end 
  legend('Stim R','Stim L','No Go','Wrong')
  
fname =['Brain Areas_VEP_Mean_LFP_', alldat.mouse_name(1:3),'_',alldat.date_exp,'_All Trial'];
hh=gcf;
set(hh,'name',fname,'Position',[ 643   148   822   418])
         if SveAllpics==1
               h = gcf;
               saveas(h,strcat(prefix,fname,'.fig'),'fig');
               saveas(h,strcat(prefix,fname,'.png'),'png');
               print(h,'-dpng', [prefix,fname], '-r600');
               print(h,'-depsc', [prefix,fname], '-r600');
           end
         if closefigs==1, close all,  end 
  
  
 %% Brain Area and Stim Locs Spkies
  
  clear  datinR  datinL  datinNo datinWr
  figure; 
  cc=1;gg=1;
  ff=1;
  for ii=1:4%:4; %Area
%   trl=109; %Selected Trial
  
  datinR=movmean((1/0.01)*squeeze(mean(mean(alldat.spks(Brain_areas_match_spks{ii},resp_R_cort,:),1),2)),5);
  datinL=movmean((1/0.01)*squeeze(mean(mean(alldat.spks(Brain_areas_match_spks{ii},resp_L_cort,:),1),2)),5);
  datinNo=movmean((1/0.01)*squeeze(mean(mean(alldat.spks(Brain_areas_match_spks{ii},resp_No_cort,:),1),2)),5);
  datinWr=movmean((1/0.01)*squeeze(mean(mean(alldat.spks(Brain_areas_match_spks{ii},resp_Al_wrong,:),1),2)),5);
  
  

      datplot1=[datinR,datinL,datinNo,datinWr];
      datplot2=[cumsum(diff(datinR).^2),cumsum(diff(datinL).^2),cumsum(diff(datinNo).^2),cumsum(diff(datinWr).^2)];


  subplot(size(datplot1,2) ,2 ,cc); cc=cc+1;
  h=plot(trial_t,datplot1,'b');
  set(h, {'color'}, num2cell(lines( size(datplot1,2) ),2));  
  xlabel(['Time(ms)'])
  ylabel(['Firing Rate'])
  title([Brain_group_global{ii}, '-All Correct Trials-Spks'])
  
  subplot(size(datplot1,2) ,2 ,cc); cc=cc+1;
  h=plot(trial_t(1:end-1),datplot2,'b');
  set(h, {'color'}, num2cell(lines( size(datplot2,2) ),2));  
  xlabel(['Time(ms)'])
  ylabel(['Cumulative Voltage'])
  title([Brain_group_global{ii}, '-All Correct Trials-Spks'])
    
  end 
  legend('Stim R','Stim L','No Go','Wrong')
  
fname =['Brain Areas_VEP_Mean_Spks_', alldat.mouse_name(1:3),'_',alldat.date_exp,'_All Trial'];
hh=gcf;
set(hh,'name',fname,'Position',[ 643   148   822   418])
         if SveAllpics==1
               h = gcf;
               saveas(h,strcat(prefix,fname,'.fig'),'fig');
               saveas(h,strcat(prefix,fname,'.png'),'png');
               print(h,'-dpng', [prefix,fname], '-r600');
               print(h,'-depsc', [prefix,fname], '-r600');
           end
         if closefigs==1, close all,  end 
  
  
  
  
  
%%  
  trl_limit=[1:50+40];
  
clear  datinR  datinL  datinNo All_PC_scores datinWr
for ii=1:4 %Loop over brain area
  datinR(:,ii)=movmean(squeeze(mean(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_R_cort, trl_limit),1),2)),5);
  datinL(:,ii)=movmean(squeeze(mean(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_L_cort, trl_limit),1),2)),5);
  datinNo(:,ii)=movmean(squeeze(mean(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_No_cort, trl_limit),1),2)),5);
  datinWr(:,ii)=movmean(squeeze(mean(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_Al_wrong, trl_limit),1),2)),5);  
end 


for mm=1:4 % Right Left NoGo Wrong
    clear datin
    if mm==1
        datin=datinR;
        answ='Right More';
    elseif mm==2
        datin=datinL;
        answ='Left More';
    elseif mm==3
        datin=datinNo;
        answ='Equal Stim';
    elseif mm==4
        datin=datinWr;
        answ='Wrong';
    end
        
      [coeff,score,latent,tsquared,explained,mu] = pca(datin-mean(datin));
      
      All_PC_scores(mm,:,:)=score';
      figure; 
      subplot(2,2,1)
      for ii=1:4
          plot(1:length(score), score(:,ii)),hold on; 
      end
      title([answ,'-All PCs-0 to 40 ms-LFP']) 
      xlabel(['Samples'])
      ylabel(['Amplitude'])
      axis tight
      legend('PC1','PC2','PC3','PC4')
%       figure; 
      subplot(2,2,2)
      clr=jet(length(score(:,1)));
        plot3(score(:,1), score(:,2),score(:,3))
        cla
        patch([score(:,1)],[score(:,2)],[score(:,3)],[1:length(score(:,1))],'EdgeColor','interp','FaceColor','none');
        colorbar('eastoutside')
      title([answ,'-Top 3 PCs-0 to 40 ms']) 
      xlabel(['PC 1'])
      ylabel(['PC 2'])
      zlabel(['PC 3'])
      
      
%       figure
      subplot(2,2,3)
      for ii=1:4
      [pxx, ff]=pwelch(score(:,ii), [],[],[],100);
      plot(ff,db(pxx))
      hold on
      end 
      title([answ,'-FFT of PCs']) 
      xlabel(['Frequency (Hz)'])
      ylabel(['PSD (dB)'])
      legend('PC1','PC2','PC3','PC4')
      axis tight
      
%       figure; 
      subplot(2,2,4)
      bar(explained)
      title([answ,'-Variance Explained']) 
      xlabel(['PC'])
      ylabel(['Variance Explained (%)'])
      ylim([0 100])
      
      
    fname =['Brain Areas_PCA_LFP_',answ,'_', alldat.mouse_name(1:3),'_',alldat.date_exp,'_All Trial'];
    hh=gcf;
    set(hh,'name',fname,'Position',[ 643   148   822   418])
     if SveAllpics==1
           h = gcf;
           saveas(h,strcat(prefix,fname,'.fig'),'fig');
           saveas(h,strcat(prefix,fname,'.png'),'png');
           print(h,'-dpng', [prefix,fname], '-r600');
           print(h,'-depsc', [prefix,fname], '-r600');
       end
     if closefigs==1, close all,  end 

      
      
      
      
      figure; 
      imagesc(datin');
      title([answ,'-Mean LFP Trials By Area']) 
      set(gca,'YTick',[1:4],'YTickLabels',{Brain_group_global{1:4}})
      colormap redblue
      
    fname =['Brain Areas_PCA_LFP_Colormap_',answ,'_', alldat.mouse_name(1:3),'_',alldat.date_exp,'_All Trial'];
    hh=gcf;
    set(hh,'name',fname,'Position',[ 643   148   822   418])
     if SveAllpics==1
           h = gcf;
           saveas(h,strcat(prefix,fname,'.fig'),'fig');
           saveas(h,strcat(prefix,fname,'.png'),'png');
           print(h,'-dpng', [prefix,fname], '-r600');
           print(h,'-depsc', [prefix,fname], '-r600');
       end
     if closefigs==1, close all,  end 
      
end 


figure
for pc=1:4 %Its not area its PCs
    subplot(2,2,pc)
      for ii=1:3
      [pxx, ff]=pwelch(squeeze(All_PC_scores(ii,pc,:)),[],[],[],100);
      plot(ff,db(pxx))
      hold on
      end 
      ylim([-120 20])
end 
legend('PC1','PC2','PC3','PC4')
    fname =['Brain Areas_PCA_lfp_fft','_', alldat.mouse_name(1:3),'_',alldat.date_exp,'_All Trial'];
    hh=gcf;
    set(hh,'name',fname,'Position',[ 643   148   822   418])
     if SveAllpics==1
           h = gcf;
           saveas(h,strcat(prefix,fname,'.fig'),'fig');
           saveas(h,strcat(prefix,fname,'.png'),'png');
           print(h,'-dpng', [prefix,fname], '-r600');
           print(h,'-depsc', [prefix,fname], '-r600');
       end
     if closefigs==1, close all,  end 



%%


  
clear  datinR  datinL  datinNo All_PC_scores datinWr
for ii=1:4 %Loop over brain area
  datinR(:,ii)=movmean((1/0.01)*squeeze(mean(mean(alldat.spks(Brain_areas_match_spks{ii},resp_R_cort,:),1),2)),5);
  datinL(:,ii)=movmean((1/0.01)*squeeze(mean(mean(alldat.spks(Brain_areas_match_spks{ii},resp_L_cort,:),1),2)),5);
  datinNo(:,ii)=movmean((1/0.01)*squeeze(mean(mean(alldat.spks(Brain_areas_match_spks{ii},resp_No_cort,:),1),2)),5);
  datinWr(:,ii)=movmean((1/0.01)*squeeze(mean(mean(alldat.spks(Brain_areas_match_spks{ii},resp_Al_wrong,:),1),2)),5);  
end 


for mm=1:4 % Right Left NoGo
    clear datin
    if mm==1
        datin=datinR;
        answ='Right More';
    elseif mm==2
        datin=datinL;
        answ='Left More';
    elseif mm==3
        datin=datinNo;
        answ='Equal Stim';
    elseif mm==4
        datin=datinWr;
        answ='Wrong';        
    end
        
      [coeff,score,latent,tsquared,explained,mu] = pca(datin-mean(datin));
      
      All_PC_scores(mm,:,:)=score';
      figure; 
      subplot(2,2,1)
      for ii=1:4
          plot(1:length(score), score(:,ii)),hold on; 
      end
      title([answ,'-All PCs-0 to 40 ms-Spks']) 
      xlabel(['Samples'])
      ylabel(['Amplitude'])
      legend('PC1','PC2','PC3','PC4')
      axis tight
      
%       figure; 
      subplot(2,2,2)
      clr=jet(length(score(:,1)));
        plot3(score(:,1), score(:,2),score(:,3))
        cla
        patch([score(:,1)],[score(:,2)],[score(:,3)],[1:length(score(:,1))],'EdgeColor','interp','FaceColor','none');
        colorbar('eastoutside')
      title([answ,'-Top 3 PCs-0 to 40 ms']) 
      xlabel(['PC 1'])
      ylabel(['PC 2'])
      zlabel(['PC 3'])
      
      
%       figure
      subplot(2,2,3)
      for ii=1:4
      [pxx, ff]=pwelch(score(:,ii), [],[],[],100);
      plot(ff,db(pxx))
      hold on
      end 
      title([answ,'-FFT of PCs']) 
      xlabel(['Frequency (Hz)'])
      ylabel(['PSD (dB)'])
      legend('PC1','PC2','PC3','PC4')
      axis tight
      
%       figure; 
      subplot(2,2,4)
      bar(explained)
      title([answ,'-Variance Explained']) 
      xlabel(['PC'])
      ylabel(['Variance Explained (%)'])
      ylim([0 100])
      
        fname =['Brain Areas_PCA_Spks_',answ,'_', alldat.mouse_name(1:3),'_',alldat.date_exp,'_All Trial'];
        hh=gcf;
        set(hh,'name',fname,'Position',[ 643   148   822   418])
         if SveAllpics==1
               h = gcf;
               saveas(h,strcat(prefix,fname,'.fig'),'fig');
               saveas(h,strcat(prefix,fname,'.png'),'png');
               print(h,'-dpng', [prefix,fname], '-r600');
               print(h,'-depsc', [prefix,fname], '-r600');
           end
         if closefigs==1, close all,  end   
         
         
      figure; 
      imagesc(datin');
      title([answ,'-Mean Spks Trials By Area']) 
      set(gca,'YTick',[1:4],'YTickLabels',{Brain_group_global{1:4}})
      colormap redblue
      
    fname =['Brain Areas_PCA_Spks_Colormap_',answ,'_', alldat.mouse_name(1:3),'_',alldat.date_exp,'_All Trial'];
    hh=gcf;
    set(hh,'name',fname,'Position',[ 643   148   822   418])
     if SveAllpics==1
           h = gcf;
           saveas(h,strcat(prefix,fname,'.fig'),'fig');
           saveas(h,strcat(prefix,fname,'.png'),'png');
           print(h,'-dpng', [prefix,fname], '-r600');
           print(h,'-depsc', [prefix,fname], '-r600');
       end
     if closefigs==1, close all,  end 
end 


figure
for pc=1:4 %Its not area its PCs
    subplot(2,2,pc)
      for ii=1:3
      [pxx, ff]=pwelch(squeeze(All_PC_scores(ii,pc,:)),[],[],[],100);
      plot(ff,db(pxx))
      hold on
      end 
      ylim([-120 20])
end 
    fname =['Brain Areas_PCA_Spks_fft','_', alldat.mouse_name(1:3),'_',alldat.date_exp,'_All Trial'];
    hh=gcf;
    set(hh,'name',fname,'Position',[ 643   148   822   418])
     if SveAllpics==1
           h = gcf;
           saveas(h,strcat(prefix,fname,'.fig'),'fig');
           saveas(h,strcat(prefix,fname,'.png'),'png');
           print(h,'-dpng', [prefix,fname], '-r600');
           print(h,'-depsc', [prefix,fname], '-r600');
       end
     if closefigs==1, close all,  end 


%% Pulling out features

close all

if pull_out_feat==1


clear datin_all lfp_ampl_late lfp_ampl_early lfp_laten_early lfp_laten_late...
    lfp_cum_sum  spks_ampl_early spks_laten_early spks_ampl_late  spks_laten_late...
    mouse_resp_time_postcue mouse_resp_cat spks_cum_sum lfp_cum_sum

w = gausswin(7); 
tpts_lfp={[5:25]+50; [15:35]+50}; %LFP time chunks to look for peak
tpts_spks={[2:35]+50};%Spike time chunks to look for peak


for dat_type=1:2
for trl=1:size(alldat.lfp,2)
 clear datin_all   
    trl_limit=[1:50+floor(alldat.gocue(trl)*100)]; %This is the number of samples to take for each trial. 
    
for ii=1:4
    clear dd
    if  dat_type==1
        %LFP
        dd=movmean(squeeze(mean(alldat.lfp(Brain_areas_match_lfp{ii},trl,trl_limit),1)),5);
    else 
        %Spikes
        dd=movmean((1/0.01)*squeeze(mean(alldat.spks(Brain_areas_match_spks{ii},trl,trl_limit),1)),5);
    end 

datin_all(:,ii)=dd;

end 

%Now that we have the 4 brain regions lets do PCA
% [coeff,score,latent,tsquared,explained,mu]
[~,score,~,~,~,~] = pca(datin_all-mean(datin_all));


%Now that we have the PCs lets pull out info

if  dat_type==1
    
    %Write target matrix
    if resp_Al_wrong(trl)==0%If its not a wrong trial
        mouse_resp_cat(trl,:)=alldat.response(trl);
        
    else
        mouse_resp_cat(trl,:)=-2; %Encode this for all wrong trials
        
    end
    mouse_resp_time_postcue(trl,:)=(alldat.response_time(trl)-alldat.gocue(trl))*100;
    
    %Info for LFP
    for pc=1:size(score,2)
            
            %Loop over brain areas (NOT PCS) and pull out the cumulative
            %sum
            cc=cumsum(diff(datin_all(:,pc)).^2);
            lfp_cum_sum(trl,pc)=cc(end);

            for tt=1:size(tpts_lfp,1) %index time periods to look at the evoked potentials
                clear sig_in; 
                sig_in=filtfilt(w, 1, sqrt(score(tpts_lfp{tt},pc).^2));
                
                [~, locs]= findpeaks( sig_in ,tpts_lfp{tt} ,...
                    'MinPeakWidth',2.5,'Annotate','extents');

                if tt==1 %Collect early compt
                    if ~isempty(locs)
                        lfp_ampl_early(trl,pc)=score(locs(1),pc);
                        lfp_laten_early(trl,pc)=locs(1);
                    else %Incase no peaks found
                        [~,mx_ind]=max(sig_in);
                        lfp_ampl_early(trl,pc)=score(tpts_lfp{tt}(mx_ind),pc);
                        lfp_laten_early(trl,pc)=tpts_lfp{tt}(mx_ind);
                    end
                else      %Collect Late compt
                    if ~isempty(locs)
                        lfp_ampl_late(trl,pc)=score(locs(1),pc);
                        lfp_laten_late(trl,pc)=locs(1);
                    else %Incase no peaks found
                        [~,mx_ind]=max(sig_in);
                        lfp_ampl_late(trl,pc)=score(tpts_lfp{tt}(mx_ind),pc);
                        lfp_laten_late(trl,pc)=tpts_lfp{tt}(mx_ind);
                    end
                end

            end

    end 


else
%Info for Spikes 

    for pc=1:size(score,2)
            
            %Loop over brain areas (NOT PCS) and pull out the cumulative
            %sum
            cc=cumsum(diff(datin_all(:,pc)).^2);
            spks_cum_sum(trl,pc)=cc(end);

            for tt=1:size(tpts_spks,1) %index time periods to look at the evoked potentials
                    clear sig_in; 
                    sig_in=filtfilt(w, 1, sqrt(score(tpts_spks{tt},pc).^2));
                [~, locs]= findpeaks( sig_in ,tpts_spks{tt} ,...
                    'MinPeakWidth',3,'Annotate','extents');

                if tt==1 %Early spikes
                    
                    if ~isempty(locs)
                        spks_ampl_early(trl,pc)=score(locs(1),pc);
                        spks_laten_early(trl,pc)=locs(1);
                    else %If no peaks found
                        [~,mx_ind]=max(sig_in);
                        spks_ampl_early(trl,pc)=score(tpts_spks{tt}(mx_ind),pc);
                        spks_laten_early(trl,pc)=tpts_spks{tt}(mx_ind);
                    end
                else %Late spikes 
                    spks_ampl_late(trl,pc)=score(locs(1),pc);
                    spks_laten_late(trl,pc)=locs(1);
                end

            end

    end 


end 



end %End of Trial loop
end % End of data type


All_feat_name={'lfp_ampl_early_PC1','lfp_ampl_early_PC2','lfp_ampl_early_PC3','lfp_ampl_early_PC4',...
               'lfp_laten_early_PC1','lfp_laten_early_PC2','lfp_laten_early_PC3','lfp_laten_early_PC4',...
               'lfp_ampl_late_PC1','lfp_ampl_late_PC2','lfp_ampl_late_PC3','lfp_ampl_late_PC4',...
               'lfp_laten_late_PC1','lfp_laten_late_PC2','lfp_laten_late_PC3','lfp_laten_late_PC4',...
               'spks_ampl_early_PC1','spks_ampl_early_PC2','spks_ampl_early_PC3','spks_ampl_early_PC4',...
               'spks_laten_early_PC1','spks_laten_early_PC2','spks_laten_early_PC3','spks_laten_early_PC4',...
               'lfp_cum_sum_PC1','lfp_cum_sum_PC2','lfp_cum_sum_PC3','lfp_cum_sum_PC4',...
              'spks_cum_sum_PC1','spks_cum_sum_PC2','spks_cum_sum_PC3','spks_cum_sum_PC4',...
              'mouse_resp_time_postcue',...
              'mouse_resp_cat'};
clear All_feat         
All_feat=[lfp_ampl_early,lfp_laten_early,lfp_ampl_late,lfp_laten_late,...
              spks_ampl_early,spks_laten_early,...
              lfp_cum_sum,...
              spks_cum_sum,...
              mouse_resp_time_postcue,...
              mouse_resp_cat];

          
          
         
          
          
          
          if select_mouse==1
              All_feat_cat=[ All_feat];
          else
              All_feat_cat=[All_feat_cat; All_feat];
          end
end        
          
end  %End of Select Mouse
%%



    if pull_out_feat==1
    %Save the feature matrix
    save('NM_Steinmetz_Features','All_feat_cat','All_feat_name','-v7')
    end 





return 






%% JUNK


%{
figure
patch(score(:,1),score(:,2),score(:,2),'EdgeColor','interp','Marker','o','MarkerFaceColor','flat');
colorbar;
      
%   legend(Brain_group_global(1:3))
      hold on
      
      %Plot trial start
      line_max=(max(datin)*2);
      line([0, 0],[-1,1]*line_max,'color','k')
      
      
      %Plot go-cue
      gocue_t=alldat.gocue(trl)*100;
      line([1 1]*gocue_t,[-1,1]*line_max,'color','r')
      
      %Plot response time
      resp_t=alldat.response_time(trl)*100;
      line([1 1]*resp_t,[-1,1]*line_max,'color','m')
      
      %Plot response time
      feedbk_t=alldat.feedback_time(trl)*100;
      line([1 1]*feedbk_t,[-1,1]*line_max,'color','g')      
      
      
      
      

      %%_______________________PCA ________________________________________
      
      resp_L_cort=logical((alldat.response==1).*(alldat.contrast_right<alldat.contrast_left))
      resp_No_cort=logical((alldat.response==0).*(alldat.contrast_right==alldat.contrast_left))
      resp_R_cort=logical((alldat.response==-1).*(alldat.contrast_right>alldat.contrast_left))
      
      dat_pca=movmean(squeeze(mean(alldat.lfp(1:12,resp_L_cort,50:end),2))',5)
      
      [coeff,score,latent,tsquared,explained,mu] = pca(dat_pca)
      
      figure; for ii=1:12, plot(1:length(score), score(:,ii)),hold on; end
      
      figure
      for ii=1:12
      [pxx, ff]=pwelch(score(:,ii), [],[],[],100)
      plot(ff,db(pxx))
      hold on
      end 
      
      figure; bar(explained)
      
      
  figure; 
 for ii=1:3, 
     plot(trial_t,squeeze(mean(alldat.pupil(ii,:,:),2))), 
     hold on, 
 end


figure; 
for ii=1:4
plot(trial_t,score(:,ii))
hold on
end 





w = gausswin(7); 
figure; 
plot(score(tpts_lfp{tm_period}+50,1))
hold on
plot(sqrt(score(tpts_lfp{tm_period}+50,1).^2))
hold on
plot(filtfilt(w, 1,sqrt(score(tpts_lfp{tm_period}+50,1).^2)))
hold on
plot(filter(w, 1,sqrt(score(tpts_lfp{tm_period}+50,1).^2)))

findpeaks(filter(w, 1,sqrt(score(tpts_lfp{tm_period}+50,1).^2)),'MinPeakWidth',3,'Annotate','extents')





C = jet(length(squeeze(All_PC_scores(nn,1,:))))%[0 .9 .75; 1 0 0; 0 0.4 0.4; 0.6 0.4 0];
clr=[jet(4)];
tt=[50]
figure;
for nn=1:4 
    scatter3(max(squeeze(All_PC_scores(nn,1,50:end))),max(squeeze(All_PC_scores(nn,2,50:end))),max(squeeze(All_PC_scores(nn,3,50:end))),[],'MarkerEdgeColor',[clr(nn,:)]); 
    hold on; 
end 







clear datinR_spks datinL_spks datinNo_spks datinWr_spks ...
      datinR_lfp datinL_lfp datinNo_lfp datinWr_lfp

for ii=1:4 %Loop over brain area
  datinR_spks(:,ii)=movmean((1/0.01)*squeeze(mean(alldat.spks(Brain_areas_match_spks{ii},resp_R_cort,:),1)),5);
  datinL_spks(:,ii)=movmean((1/0.01)*squeeze(mean(alldat.spks(Brain_areas_match_spks{ii},resp_L_cort,:),1)),5);
  datinNo_spks(:,ii)=movmean((1/0.01)*squeeze(mean(alldat.spks(Brain_areas_match_spks{ii},resp_No_cort,:),1)),5);
  datinWr_spks(:,ii)=movmean((1/0.01)*squeeze(mean(alldat.spks(Brain_areas_match_spks{ii},resp_Al_wrong,:),1)),5);    
  
  
  datinR_lfp(:,ii)=movmean(squeeze(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_R_cort,:),1)),5);
  datinL_lfp(:,ii)=movmean(squeeze(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_L_cort,:),1)),5);
  datinNo_lfp(:,ii)=movmean(squeeze(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_No_cort,:),1)),5);
  datinWr_lfp(:,ii)=movmean(squeeze(mean(alldat.lfp(Brain_areas_match_lfp{ii},resp_Al_wrong,:),1)),5);   
  
end 


figure;hold on
trl=1
plot(Test(trl,:))

[num, ind]=max(Test(trl,50:50+20))
plot(ind+49, num,'r*')

[num2, ind2]=max(Test(trl,50+20:50+50))
plot(ind2+(49+20), num2,'r*')


%}