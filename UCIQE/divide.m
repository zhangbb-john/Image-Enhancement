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
count=0;
ifdivide=0;
if ifdivide==1
    while ii<=len
        if(ii==14)
            'error'
        end
        imgpath = fullfile( maindir2,   imgdat2( ii).name);
        img=imread(imgpath);
        for p=1:1
            for q=1:1
                patch=img(int32((p-1)/3*size(img,1)+1):int32(p/3*size(img,1)),...
                    int32((p-1)/3*size(img,2)+1):int32(p/3*size(img,2)),:);
                imwrite(patch,['E:\desktop\recent_files\graduate_design\myfiles',...
                    '\03reading_paper\metrics\0209An Underwater Colour Image Quality\prj1\1\',num2str(count,'%.4d'),...
                    '.jpg']);
                count=count+1;
            end
        end
        imgpath = fullfile( maindir,   imgdat( ii).name);   
        img=imread(imgpath);
        for p=1:1
            for q=1:1
                patch=img(int32((p-1)/3*size(img,1)+1):int32(p/3*size(img,1)),...
                    int32((p-1)/3*size(img,2)+1):int32(p/3*size(img,2)),:);
                imwrite(patch,['E:\desktop\recent_files\graduate_design\myfiles',...
                    '\03reading_paper\metrics\0209An Underwater Colour Image Quality\prj1\1\',num2str(count,'%.4d'),...
                    '.jpg']);
                count=count+1;
            end
        end
        imgpath = fullfile( maindir3,   imgdat3( ii).name);
        img=imread(imgpath);
        for p=1:1
            for q=1:1
                patch=img(int32((p-1)/3*size(img,1)+1):int32(p/3*size(img,1)),...
                    int32((p-1)/3*size(img,2)+1):int32(p/3*size(img,2)),:);
                imwrite(patch,['E:\desktop\recent_files\graduate_design\myfiles',...
                    '\03reading_paper\metrics\0209An Underwater Colour Image Quality\prj1\1\',num2str(count,'%.4d'),...
                    '.jpg']);
                count=count+1;
            end
        end
        imgpath = fullfile( maindir4,   imgdat4( ii).name);
        img=imread(imgpath);
        for p=1:1
            for q=1:1
                patch=img(int32((p-1)/3*size(img,1)+1):int32(p/3*size(img,1)),...
                    int32((p-1)/3*size(img,2)+1):int32(p/3*size(img,2)),:);
                imwrite(patch,['E:\desktop\recent_files\graduate_design\myfiles',...
                    '\03reading_paper\metrics\0209An Underwater Colour Image Quality\prj1\1\',num2str(count,'%.4d'),...
                    '.jpg']);
                count=count+1;
            end
        end
        ii=ii+1;
    end
else 
    while ii<=len
        if(ii==14)
            'error'
        end
        imgpath = fullfile( maindir2,   imgdat2( ii).name);
        img=imread(imgpath);
        for p=1:1
            for q=1:1
                patch=img(int32((p-1) *size(img,1)+1):int32(p *size(img,1)),...
                    int32((p-1) *size(img,2)+1):int32(p *size(img,2)),:);
                imwrite(patch,['E:\desktop\recent_files\graduate_design\myfiles',...
                    '\03reading_paper\metrics\0209An Underwater Colour Image Quality\prj1\1\',num2str(count  ,'%.4d'),...
                    '.jpg']);
                count=count+1;
            end
        end
        imgpath = fullfile( maindir,   imgdat( ii).name);   
        img=imread(imgpath);
        for p=1:1
            for q=1:1
                patch=img(int32((p-1) *size(img,1)+1):int32(p *size(img,1)),...
                    int32((p-1) *size(img,2)+1):int32(p *size(img,2)),:);
                imwrite(patch,['E:\desktop\recent_files\graduate_design\myfiles',...
                    '\03reading_paper\metrics\0209An Underwater Colour Image Quality\prj1\1\',num2str(count  ,'%.4d'),...
                    '.jpg']);
                count=count+1;
            end
        end
        imgpath = fullfile( maindir3,   imgdat3( ii).name);
        img=imread(imgpath);
        for p=1:1
            for q=1:1
                patch=img(int32((p-1) *size(img,1)+1):int32(p *size(img,1)),...
                    int32((p-1) *size(img,2)+1):int32(p *size(img,2)),:);
                imwrite(patch,['E:\desktop\recent_files\graduate_design\myfiles',...
                    '\03reading_paper\metrics\0209An Underwater Colour Image Quality\prj1\1\',num2str(count  ,'%.4d'),...
                    '.jpg']);
                count=count+1;
            end
        end
        imgpath = fullfile( maindir4,   imgdat4( ii).name);
        img=imread(imgpath);
        for p=1:1
            for q=1:1
                patch=img(int32((p-1) *size(img,1)+1):int32(p *size(img,1)),...
                    int32((p-1) *size(img,2)+1):int32(p *size(img,2)),:);
                imwrite(patch,['E:\desktop\recent_files\graduate_design\myfiles',...
                    '\03reading_paper\metrics\0209An Underwater Colour Image Quality\prj1\1\',num2str(count  ,'%.4d'),...
                    '.jpg']);
                count=count+1;
            end
        end
        ii=ii+1;
    end
end
    
