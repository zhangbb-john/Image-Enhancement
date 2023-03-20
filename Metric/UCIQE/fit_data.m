
clc;
close all;
maindir = 'E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\metrics\0209An Underwater Colour Image Quality\prj1\1\1';
% score=[5,2,3,1.50000000000000,3,2,3,2,1.50000000000000,1.50000000000000,2.50000000000000,2.50000000000000,...
%     2.50000000000000,2,1.50000000000000,3,1.50000000000000,2,3,2.50000000000000,2,3,4,2.50000000000000,3,...
%     2.50000000000000,5,3.50000000000000,3,3,2.50000000000000,2.50000000000000,1.50000000000000,2.50000000000000,...
%     3.50000000000000,3,4,3,2.50000000000000,3,2.50000000000000,2.50000000000000,3,2,3,4.50000000000000,...
%     2.50000000000000,2.50000000000000,3,3.50000000000000];
subdir  = dir( maindir );
scores=[
5	5	2	4	3	4	5	4	5	5	4	3	5	3	5	4	3	4	3	3	4	5	5	5	5	5	5	5	4	5	5	4	5	4
2	2	2	4	2	3	5	3	5	4	4	2	3	3	5	4	4	4	3	4	4	5	5	5	4	5	5	5	4	5	4	4	4	3
3	3	3	5	2	2	4	2	4	1	3	2	2	3	3	3	4	3	2	4	3	4	3	3	3	4	3	1	3	4	2	2	3	3
2	1	1	3	3	2	3	2	3	4	2	3	5	2	4	3	4	3	2	4	3	4	4	3	5	3	3	5	3	4	3	3	4	3
3	3	3	5	3	2	4	2	4	5	3	3	5	2	5	2	4	4	3	3	4	3	4	3	3	4	4	3	4	4	3	4	3	4
2	2	2	4	2	2	4	2	4	5	3	2	3	3	4	4	3	3	3	3	4	4	4	3	3	4	4	2	2	4	3	3	2	3
4	2	2	5	1	3	4	3	4	5	2	1	1	3	5	4	4	4	4	3	5	4	3	3	2	3	3	2	1	5	2	4	2	4
3	1	3	5	2	2	5	2	5	5	3	2	5	3	4	4	4	4	3	3	5	4	5	3	3	4	4	2	3	5	4	2	2	4
2	1	1	2	2	3	4	3	4	5	2	2	4	2	4	2	4	4	4	2	4	3	4	3	3	4	4	3	2	4	3	2	2	3
2	1	1	2	1	1	3	1	3	2	3	1	1	2	4	3	3	3	3	2	2	3	2	2	2	3	2	1	1	4	2	2	1	3
2	3	2	4	2	3	4	3	4	4	3	2	2	2	5	2	2	4	4	3	3	3	4	4	4	4	3	2	3	4	3	2	5	4
3	2	2	5	3	3	4	3	4	5	3	3	5	4	4	3	3	4	4	3	3	4	4	3	4	4	3	4	4	4	4	4	3	4
3	2	1	4	2	2	4	2	4	1	2	2	2	3	3	2	5	2	3	4	3	5	3	3	1	4	3	1	2	4	1	2	5	4
2	2	1	3	1	2	4	2	4	2	2	1	1	2	2	2	4	2	2	3	2	3	2	2	2	4	2	1	3	4	1	1	5	3
2	1	1	3	1	1	4	1	4	3	2	1	1	2	3	2	2	3	3	2	3	4	4	2	3	3	3	1	3	3	2	1	3	3
4	2	1	4	1	3	5	3	5	3	2	1	4	3	4	4	4	3	4	3	4	2	4	3	3	3	3	2	4	4	2	2	2	4
2	1	2	2	1	2	4	2	4	4	2	1	4	3	4	3	3	2	3	2	2	2	3	2	4	3	3	2	3	4	2	3	2	3
3	1	2	3	2	2	4	2	4	4	3	2	5	4	4	5	4	3	4	1	3	4	3	2	4	5	3	1	3	3	3	4	5	3
3	3	2	5	2	3	4	3	4	5	3	2	4	4	3	4	4	4	4	2	3	3	4	3	4	5	4	4	2	4	3	3	3	4
3	2	1	4	2	2	3	2	3	3	3	2	5	4	4	3	5	3	5	2	4	4	3	3	3	4	3	2	2	4	3	2	5	4
2	2	2	4	3	2	4	2	4	4	3	3	5	3	4	5	5	3	5	1	4	3	4	4	5	5	5	3	4	5	4	3	2	4
3	3	1	3	3	2	4	2	4	5	3	3	5	3	4	4	5	3	5	1	3	2	5	4	4	4	3	2	1	4	4	4	3	4
4	4	2	5	3	3	5	3	5	3	3	3	3	3	5	4	5	2	3	4	4	5	4	4	5	4	4	3	1	4	4	3	4	4
3	2	2	3	3	2	4	2	4	4	3	3	4	3	5	3	2	3	4	3	3	4	5	3	5	5	4	4	4	4	4	3	4	4
3	3	1	3	2	3	4	3	4	4	2	2	5	3	3	3	3	3	4	1	2	3	4	2	4	3	3	2	1	4	2	2	4	3
2	3	2	5	3	2	4	2	4	5	4	3	5	5	5	4	3	4	4	3	3	5	5	4	5	5	4	5	3	5	4	4	5	4
5	5	2	5	4	3	5	3	5	5	3	4	5	3	5	4	5	4	5	4	4	5	5	5	5	5	4	5	3	3	5	4	5	4
3	4	1	5	4	3	4	3	4	5	3	4	4	3	4	5	4	4	5	4	3	4	5	4	5	5	3	5	3	4	5	3	4	4
3	3	2	5	4	3	5	3	5	5	5	4	4	4	5	5	5	4	5	4	3	4	5	5	4	5	4	5	4	5	4	4	3	4
4	2	2	4	3	2	3	2	3	4	4	3	2	4	4	3	3	3	3	2	2	3	5	3	4	4	3	4	2	2	3	1	2	4
3	2	2	4	2	2	4	2	4	3	3	2	1	3	4	2	4	2	3	2	3	4	4	4	3	4	4	3	4	2	3	1	4	3
2	3	2	3	3	2	4	2	4	3	3	3	4	3	4	4	3	3	5	3	2	4	4	4	3	4	4	5	3	3	4	3	4	4
2	1	1	2	1	1	4	1	4	1	3	1	1	2	3	2	2	2	2	1	2	3	3	2	3	3	1	1	1	1	3	1	2	2
3	2	2	5	3	2	3	2	3	3	3	3	3	4	4	3	4	3	4	3	2	4	4	3	4	4	4	4	4	3	4	2	4	4
3	4	2	5	4	3	4	3	4	5	3	4	5	3	5	4	5	3	5	3	4	5	5	5	5	5	4	5	3	4	4	3	5	4
4	2	2	2	3	2	4	2	4	3	3	3	4	2	4	4	2	2	2	4	2	4	3	3	4	4	3	5	1	3	2	3	5	3
4	4	2	4	3	3	4	3	4	3	3	3	3	2	4	2	5	2	2	4	3	5	3	4	5	4	4	5	4	3	3	4	5	4
3	3	1	3	4	2	5	2	5	4	3	4	5	3	5	5	3	3	5	2	1	3	4	5	5	4	4	5	3	4	4	4	5	3
4	1	1	3	2	2	4	2	4	4	3	2	5	2	4	3	2	3	4	3	2	4	4	4	5	4	3	4	1	4	3	2	5	4
3	3	2	5	5	3	4	3	4	5	4	5	5	2	5	1	5	2	4	3	3	5	4	4	4	5	4	5	5	3	4	1	5	3
3	2	1	3	4	2	5	2	5	2	4	4	4	3	5	3	3	2	5	1	1	2	5	2	4	3	3	4	2	3	5	2	3	4
2	3	2	5	3	4	5	4	5	4	4	3	4	3	4	5	5	3	4	4	5	5	4	5	5	5	3	4	4	4	3	1	3	4
3	3	1	2	3	3	4	3	4	2	3	3	5	2	4	3	2	2	2	3	1	2	2	2	5	4	2	1	1	4	4	3	5	3
2	2	1	2	2	1	4	1	4	1	3	2	2	2	3	5	2	2	3	2	2	1	2	1	2	3	1	1	2	3	3	2	2	3
2	4	2	5	4	3	5	3	5	5	4	4	5	3	4	3	3	2	4	1	4	3	4	4	4	4	4	5	2	4	4	3	5	4
4	5	2	5	4	3	5	3	5	4	5	4	5	2	5	4	4	3	5	4	3	4	5	4	4	5	4	5	3	5	5	4	5	3
4	1	1	4	1	1	3	1	3	1	2	1	1	3	2	5	3	2	2	3	2	3	3	2	3	4	2	2	1	2	3	1	3	3
3	2	1	3	1	1	4	1	4	3	3	1	2	3	4	2	4	3	3	4	2	4	3	4	3	4	2	5	1	4	3	3	3	3
3	3	2	5	1	4	4	4	4	5	3	1	1	2	5	3	4	2	3	3	3	5	4	4	3	4	2	4	4	3	3	2	3	4
3	4	1	5	3	4	5	4	5	5	4	3	4	2	5	4	5	3	3	4	4	5	5	3	4	4	4	5	5	4	4	4	5	4


];
newscores=zeros(size(scores));
newscores(scores==5)=1;
newscores(scores==4)=2;
newscores(scores==3)=3;

newscores(scores==2)=4;
newscores(scores==1)=5;
score=mean(newscores,2);

for i = 1 : length( subdir )
    if( isequal( subdir( i ).name, '.' )||...
        isequal( subdir( i ).name, '..')||...
        ~subdir( i ).isdir)               % 如果不是目录则跳过
        continue;
    end
    subimg = fullfile( maindir, subdir( i ).name, '*.jpg' );
    imgdat = dir( subimg );               % 子文件夹下找后缀为dat的文件
    sigma_c_all=zeros(1,length(imgdat));
    contrast_l_all=zeros(1,length(imgdat));
    u_s_all=zeros(1,length(imgdat));

    for j = 1 : length( imgdat )
        
        imgpath = fullfile( maindir, subdir( i ).name, imgdat( j ).name);
        image=imread(imgpath);
        lab = rgb2lab_n(image);
        l = lab(:,:,1);l(l==0)=1;
        a = lab(:,:,2);
        b = lab(:,:,3);

        chroma = sqrt(a.^2 + b.^2);
        % average of chroma
        u_c = mean(chroma(:));
        % variance of chroma
        sigma_c = sqrt(mean(mean(chroma.^2 - u_c.^2)));

        saturation = chroma ./ l;
        % average of saturation
        u_s = mean(saturation(:));

        contrast_l = max(l(:)) - min(l(:));

        sigma_c_all(j)=sigma_c;
        contrast_l_all(j)=contrast_l;
        u_s_all(j)=u_s;
        figure
        imshow(image);
        title(num2str(score(j)));
    end   
end

    
sigma_c_all=mapminmax(sigma_c_all,0,1);
contrast_l_all=mapminmax(contrast_l_all,0,1);
u_s_all=mapminmax(u_s_all,0,1);
x=[sigma_c_all(:),contrast_l_all(:),u_s_all(:)];
a0=[0,0,0];
f=@(a,x)a(1)*x(:,1)+a(2)*x(:,2)+a(3)*x(:,3);
A=lsqcurvefit(f,a0,x,score);
y=A(1)*x(:,1)+A(2)*x(:,2)+A(3)*x(:,3);
plot(1:length(y),y,'r.');
hold on;
plot(1:length(y),score,'b.');
['误差率',num2str(mean(abs((y-score))./score))]
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
original=zeros(1,len);
histo=zeros(1,len);
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
    original(ii)=UCIQE(img,A(1),A(2),A(3));
    
    imgpath = fullfile( maindir,   imgdat( ii).name);   
	img_udcp=imread(imgpath);
    udcp(ii)=UCIQE(img_udcp,A(1),A(2),A(3));
    
    imgpath = fullfile( maindir1,   imgdat1( ii).name);      
    img_hist=imread(imgpath);
    histo(ii)=UCIQE(img_hist,A(1),A(2),A(3));
    
    imgpath = fullfile( maindir3,   imgdat3( ii).name);
	img_dcp=imread(imgpath);
    dcp(ii)=UCIQE(img_dcp,A(1),A(2),A(3));
    
    imgpath = fullfile( maindir4,   imgdat4( ii).name);
	img_dl=imread(imgpath);
    dl(ii)=UCIQE(img_dl,A(1),A(2),A(3));
    ii=ii+1;
end
figure;
plot(original,'y.');
hold on;
plot(udcp,'r.');
hold on;
plot(dcp,'g.');
hold on;
plot(dl,'b.');
hold on;
plot(histo,'k.');

title(['original:',num2str(mean(original)),'var',num2str(var(original)),...
    'udcp:',num2str(mean(udcp)),'var',num2str(var(udcp)),...
    ';histo:',num2str(mean(histo)),'var',num2str(var(histo)),....
    ';dcp:',num2str(mean(dcp)),'var',num2str(var(dcp)),....
    ';deep learning:',num2str(mean(dl)),'var',num2str(var(dl))]);
figure;
plot(1:50,score/max(score),'r');hold on;
plot(1:50,sigma_c_all/max(sigma_c_all),'g');hold on;
plot(1:50,contrast_l_all/max(contrast_l_all),'b');hold on;
plot(1:50,u_s_all/max(u_s_all),'k');hold on;

