clc;clear all;%% 读取视频
close all;

maindir='E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\mathematic model\UDCP\UDCPprj\prj2\OutputDepth';
subimg = fullfile( maindir, '*.jpg' );
imgdat = dir( subimg ); 
len=length(imgdat);

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

udcp=zeros(1,len);
dcp=zeros(1,len);
dl=zeros(1,len);
ii=1;
while ii<=len
    if(ii==14)
        'error'
    end
    imgpath = fullfile( maindir2,   imgdat2( ii).name);
    img=imread(imgpath);
    imgpath = fullfile( maindir,   imgdat( ii).name);   
	img_udcp=imread(imgpath);
    udcp(ii)=UCIQE(img_udcp);
    imgpath = fullfile( maindir3,   imgdat3( ii).name);
	img_dcp=imread(imgpath);
    dcp(ii)=UCIQE(img_dcp);
    imgpath = fullfile( maindir4,   imgdat4( ii).name);
	img_dl=imread(imgpath);
    dl(ii)=UCIQE(img_dl);
    ii=ii+1;
end
figure;
plot(udcp,'r.');
title(['udcp:',num2str(mean(udcp))]);
hold on;
plot(dcp,'g.');
hold on;
plot(dl,'b.');

title(['udcp:',num2str(mean(udcp)),';dcp:',num2str(mean(dcp)),';deep learning:',num2str(mean(dl))]);
