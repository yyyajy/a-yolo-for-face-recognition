function im_data = yolo_prepare(im)
%把matlab的图片格式转换成caffe中图片的格式,matlab读取的是H*W*C的格式
[w,h,~]=size(im);%w=H h=W
%此处构造的是 H*W的矩阵，所以才有了下面的permute
mean_data(:,:,1)=ones(w,h).*127.5;%B
mean_data(:,:,2)=ones(w,h).*127.5;%G
mean_data(:,:,3)=ones(w,h).*127.5;%R
mean_data=permute(mean_data,[2,1,3]);%变为W*H*C BGR的数据格式，而

im_data=im(:,:,[3,2,1]);%RGB to BGR  
im_data=permute(im,[2,1,3]);%旋转高度和宽度，因为caffe支持的是W*H*C,而matlab读取的是H*W*C
im_data=single(im_data);% uint8 转为single 类型
im_data=(im_data-single(mean_data))*0.00784; %减掉均值文件

