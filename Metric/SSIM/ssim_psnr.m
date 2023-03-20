maindir = 'E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\metrics\ssim\img_select\original';
subimg = fullfile( maindir,  '*.*g' );
imgdat = dir( subimg );               % 子文件夹下找后缀为dat的文件

maindirdcp = 'E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\mathematic model\DCP\prj\prj6\OutputImages';
subimg_dcp = fullfile( maindirdcp,  '*.*g' );
imgdat_dcp = dir( subimg_dcp );               % 子文件夹下找后缀为dat的文件

maindirudcp = 'E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\mathematic model\UDCP\UDCPprj\prj4\OutputDepth';
subimg_udcp = fullfile( maindirudcp,  '*.*g' );
imgdat_udcp = dir( subimg_udcp );               % 子文件夹下找后缀为dat的文件

maindirdl = 'E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\deep learning\waterGan\prj_restore\prj7h\test_result4';
subimg_dl = fullfile( maindirdl,  '*.*g' );
imgdat_dl = dir( subimg_dl );               % 子文件夹下找后缀为dat的文件
Len=min([length( imgdat ),length(imgdat_dcp),length(imgdat_udcp),length(imgdat_dl)]);



maindirHe = 'E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\optimization-based\histogram\prj4\result';
subimg_he = fullfile( maindirHe,  '*.*g' );
imgdat_he = dir( subimg_he );               % 子文件夹下找后缀为dat的文件


SSIM_DCP=zeros(1,Len);
PSNR_DCP=zeros(1,Len);
SSIM_UDCP=zeros(1,Len);
PSNR_UDCP=zeros(1,Len);
SSIM_DL=zeros(1,Len);
PSNR_DL=zeros(1,Len);
SSIM_HE=zeros(1,Len);
PSNR_HE=zeros(1,Len);

for j = 1 : Len
    imgpath = fullfile( maindir,  imgdat( j ).name);
    image=imread(imgpath);       
    imgpath = fullfile( maindirdcp,  imgdat_dcp( j ).name);
    imagedcp=imread(imgpath); 
    SSIM_DCP(j)=ssim(imagedcp,image);
    PSNR_DCP(j)=psnr(imagedcp,image);
    imgpath = fullfile( maindirudcp,  imgdat_udcp( j ).name);
    imageudcp=imread(imgpath);   
    SSIM_UDCP(j)=ssim(imageudcp,image);
    PSNR_UDCP(j)=psnr(imageudcp,image);  
    imgpath = fullfile( maindirdl,  imgdat_dl( j ).name);
    imagedl=imread(imgpath);   
    SSIM_DL(j)=ssim(imagedl,image);
    PSNR_DL(j)=psnr(imagedl,image); 
    
    imgpath = fullfile( maindirHe,  imgdat_dcp( j ).name);
    imagehe=imread(imgpath); 
    SSIM_He(j)=ssim(imagehe,image);
    PSNR_He(j)=psnr(imagehe,image);
end
['SSIM_DCP',num2str(mean(SSIM_DCP))]
['PSNR_DCP',num2str(mean(PSNR_DCP))]
['SSIM_UDCP',num2str(mean(SSIM_UDCP))]
['PSNR_UDCP',num2str(mean(PSNR_UDCP))]
['SSIM_DL',num2str(mean(SSIM_DL))]
['PSNR_DL',num2str(mean(PSNR_DL))]
['SSIM_He',num2str(mean(SSIM_He))]
['PSNR_He',num2str(mean(PSNR_He))]

