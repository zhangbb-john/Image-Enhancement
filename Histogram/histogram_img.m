maindir = 'E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\optimization-based\histogram\prj4\img';


subimg = fullfile( maindir,  '*.*g' );
imgdat = dir( subimg );               % 子文件夹下找后缀为dat的文件
for j = 1 : length( imgdat )

    imgpath = fullfile( maindir,  imgdat( j ).name);
    image=imread(imgpath);
    imwrite(histeq(image),['E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\optimization-based\histogram\prj4\result\',...
        imgdat( j ).name]); 

end

