clc;clear all;%% 读取视频
close all;

maindir='E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\mathematic model\UDCP\UDCPprj\prj2\OutputDepth';
subimg = fullfile( maindir, '*.jpg' );
imgdat = dir( subimg ); 

maindir1='E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\optimization-based\histogram\prj1\result';
subimg1 = fullfile( maindir1, '*.jpg' );
imgdat1 = dir( subimg1 ); 

maindir2='E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\mathematic model\UDCP\UDCPprj\img';
subimg2 = fullfile( maindir2, '*.jpg' );
imgdat2 = dir( subimg2 ); 
maindir3='E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\mathematic model\DCP\prj\prj2\OutputImages';
subimg3 = fullfile( maindir3, '*.jpg' );
imgdat3 = dir( subimg3 ); 

maindir4='E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\deep learning\waterGan\prj_restore\prj7f\test_result';
subimg4 = fullfile( maindir4, '*.png' );
imgdat4 = dir( subimg4 ); 
len=length(imgdat4);

Rudcp=zeros(1,len);
Eudcp=zeros(1,len);
SigmaUdcp=zeros(1,len);
Rhist=zeros(1,len);
Ehist=zeros(1,len);
SigmaHist=zeros(1,len);
Rdcp=zeros(1,len);
Edcp=zeros(1,len);
SigmaDcp=zeros(1,len);
Rdl=zeros(1,len);
Edl=zeros(1,len);
SigmaDl=zeros(1,len);
ii=1;
std=0;
while ii<=len
    imgpath = fullfile( maindir2,   imgdat2( ii).name);
    img=imread(imgpath);
    
    imgpath = fullfile( maindir,   imgdat( ii).name);   
	img_udcp=imread(imgpath);
    [Rudcp(ii),Eudcp(ii),SigmaUdcp(ii)]=tauMetrics(img,img_udcp);
    
    imgpath = fullfile( maindir1,   imgdat1( ii).name);   
	img_hist=imread(imgpath);
    [Rhist(ii),Ehist(ii),SigmaHist(ii)]=tauMetrics(img,img_hist);
    
    imgpath = fullfile( maindir3,   imgdat3( ii).name);
	img_dcp=imread(imgpath);
    [Rdcp(ii),Edcp(ii),SigmaDcp(ii)]=tauMetrics(img,img_dcp);
    
    imgpath = fullfile( maindir4,   imgdat4( ii).name);
	img_dl=imread(imgpath);
    [Rdl(ii),Edl(ii),SigmaDl(ii)]=tauMetrics(img,img_dl);
    ii=ii+1;
end
figure;
plot(Rudcp,'r.');
title(['udcp:r=',num2str(mean(Rudcp)),';dcp:r=',num2str(mean(Rdcp)),';deep learning:r=',num2str(mean(Rdl))]);
hold on;
plot(Rdcp,'g.');
hold on;
plot(Rdl,'b.');
hold on;
plot(Rhist,'y.');
figure;
plot(Eudcp,'r.');
title(['udcp:r=',num2str(mean(Eudcp)),';dcp:r=',num2str(mean(Edcp)),';deep learning:r=',num2str(mean(Edl))]);
hold on;
plot(Edcp,'g.');
hold on;
plot(Edl,'b.');
hold on;
plot(Ehist,'y.');
figure;
plot(SigmaUdcp,'r.');
title(['udcp:r=',num2str(mean(SigmaUdcp)),';dcp:r=',num2str(mean(SigmaDcp)),';deep learning:r=',num2str(mean(SigmaDl))]);
hold on;
plot(SigmaDcp,'g.');
hold on;
plot(SigmaDl,'b.');
hold on;
plot(SigmaHist,'y.');
if (std==0)
    R=[Rudcp;Rdcp;Rdl;Rhist];
    R=reshape(mapminmax(R(:)',0,1),size(R));
    Avg_R=mean(R,2);
    E=[Eudcp;Edcp;Edl;Ehist];
    E=reshape(mapminmax(E(:)',0,1),size(E));

    Avg_E=mean(E,2);
    sigma=[SigmaUdcp;SigmaDcp;SigmaDl;SigmaHist];
    sigma=reshape(mapminmax(1-sigma(:)',0,1),size(sigma));

    Avg_sigma=mean(sigma,2);
    performance=R+E+sigma;
    Avg_performance=mean(performance,2);
elseif(std==2)
    R=[Rudcp;Rdcp;Rdl;Rhist];
%     R=mapminmax(R,0,1);
    R=reshape(mapstd(R(:)'),size(R));    
    R=reshape(mapminmax(R(:)',0,1),size(R));
    Avg_R=mean(R,2);
    
    E=[Eudcp;Edcp;Edl;Ehist];
    E=reshape(mapstd(E(:)'),size(E));
    E=reshape(mapminmax(E(:)',0,1),size(E));
    Avg_E=mean(E,2);
    
    sigma=[SigmaUdcp;SigmaDcp;SigmaDl;SigmaHist];
    sigma=reshape(mapstd(1-sigma(:)'),size(sigma));
    sigma=reshape(mapminmax(sigma(:)',0,1),size(sigma));    
    Avg_sigma=mean(sigma,2);

else
    R=[Rudcp;Rdcp;Rdl;Rhist];
%     R=mapminmax(R,0,1);
    R=reshape(mapstd(R(:)'),size(R));
    Avg_R=mean(R,2);
    E=[Eudcp;Edcp;Edl;Ehist];
%     E=mapminmax(E,0,1);

    E=reshape(mapstd(E(:)'),size(E));
    Avg_E=mean(E,2);
    sigma=[SigmaUdcp;SigmaDcp;SigmaDl;SigmaHist];
    sigma=reshape(mapstd(1-sigma(:)'),size(sigma));
%     sigma=mapminmax(1-sigma,0,1);

    Avg_sigma=mean(sigma,2);
    performance=R+E+sigma;performance=reshape(mapminmax(performance(:)',0,1),size(performance));
    Avg_performance=mean(performance,2);   
end

% R=[Rudcp;Rdcp;Rdl];R=mapminmax(R);
% E=[Eudcp;Edcp;Edl];E=mapminmax(E);
% sigma=[SigmaUdcp;SigmaDcp;SigmaDl];sigma=mapminmax(1-sigma);
% performance=R+E+sigma;
% Avg_performance=mean(performance,2);
% 
% R=[Rudcp;Rdcp;Rdl];R=mapminmax(R);
% E=[Eudcp;Edcp;Edl];E=mapminmax(E);
% sigma=[SigmaUdcp;SigmaDcp;SigmaDl];sigma=mapminmax(1-sigma);
% performance=R+E;
% Avg_performance=mean(performance,2);

