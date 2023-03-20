maindir = 'E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\metrics\0209An Underwater Colour Image Quality\prj1\1';
subdir  = dir( maindir );
num=50;
for i = 1 : length( subdir )
    if( isequal( subdir( i ).name, '.' )||...
        isequal( subdir( i ).name, '..')||...
        ~subdir( i ).isdir)               % �������Ŀ¼������
        continue;
    end
    subimg = fullfile( maindir, subdir( i ).name, '*.jpg' );
    imgdat = dir( subimg );               % ���ļ������Һ�׺Ϊdat���ļ�
    num=randperm(length( imgdat ),num);

    for j = 1 : length( num )
        
        imgpath = fullfile( maindir, subdir( i ).name, imgdat( num(j) ).name);
        image=imread(imgpath);
        imwrite(image,['E:\desktop\recent_files\graduate_design\myfiles\03reading_paper\metrics\0209An Underwater Colour Image Quality\prj1\imgSelect\',...
            num2str(j,'%06d'),'.jpg']); 
        
    end
end
