function im_data = yolo_prepare(im)
%��matlab��ͼƬ��ʽת����caffe��ͼƬ�ĸ�ʽ,matlab��ȡ����H*W*C�ĸ�ʽ
[w,h,~]=size(im);%w=H h=W
%�˴�������� H*W�ľ������Բ����������permute
mean_data(:,:,1)=ones(w,h).*127.5;%B
mean_data(:,:,2)=ones(w,h).*127.5;%G
mean_data(:,:,3)=ones(w,h).*127.5;%R
mean_data=permute(mean_data,[2,1,3]);%��ΪW*H*C BGR�����ݸ�ʽ����

im_data=im(:,:,[3,2,1]);%RGB to BGR  
im_data=permute(im,[2,1,3]);%��ת�߶ȺͿ�ȣ���Ϊcaffe֧�ֵ���W*H*C,��matlab��ȡ����H*W*C
im_data=single(im_data);% uint8 תΪsingle ����
im_data=(im_data-single(mean_data))*0.00784; %������ֵ�ļ�

