clear;
addpath('..') %����+caffe·��  
addpath('.')
caffe.set_mode_cpu();%����CPUģʽ  
model = 'D:/CelebA/matlab_demo/yolo/yolo/yolo-face-deployori.prototxt';%ģ��deploy  
weights = 'D:/CelebA/matlab_demo/yolo/yolo/yolo-face.caffemodel';%����  .
%  weights = 'D:/CelebA/matlab_demo/yolo/yolo/_iter_65000.caffemodel';
net=caffe.Net(model,weights,'test');%����  
pdollar_toolbox_path='C:\caffe-master\matlab\demo\yolo';
addpath(genpath(pdollar_toolbox_path));
side=11;
B=2;
class_index=1;%��������
confidence_index=side*side+1;%���Ŷȵ�����
loc_index=confidence_index+side*side*B;
thresh = 0.25;%�趨��ֵ����������������ֵʱ����box����
bbox=[];
bbox_id=1; 
img=imread('F:\FDDB\originalPics\2003\01\24\big\img_23.jpg');%23 333 28 764 634
% img=imread(' C:\Users\Administrator\Desktop\pyramid\2\face1.jpg');
img_h=size(img,1);
img_w=size(img,2);
imshow(img);
img=imresize(img,[448 448]);
input_data={yolo_prepare(img)};
scores=net.forward(input_data);     
feat=net.blobs('fc12').get_data();%�Ի�ȡ��1331����ֵ
%�������1331����ֵ������
for ii=1:side*side%121��λ��  ���к���
   row=floor((ii-1)/side+1);
   col=ii-side*(row-1);
   class_index=ii;%ÿ�����������������
   for jj=1:B% ÿ��λ���ϵ�B����
       confidence_index=side*side+B*(ii-1)+jj; %ÿ��box�����Ŷ�����     
       pro=feat(class_index,1)*feat(confidence_index,1); % �����Ϣ*���Ŷ�    
       if pro>thresh           
           loc_index=side*side+side*side*2+1+((ii-1)*B+jj-1)*4;%�ĸ�����ֵ����ʼλ��x
           bbox(bbox_id,1) = (feat(loc_index,1)+col-1)*448/side;%X box������λ��
           bbox(bbox_id,2) = (feat(loc_index+1,1)+row-1)*448/side;%y box������λ��
           bbox(bbox_id,3) = (feat(loc_index+2,1)*feat(loc_index+2,1))*448;%w
           bbox(bbox_id,4) = (feat(loc_index+3,1)*feat(loc_index+3,1))*448;%h 
           bbox(bbox_id,5)=pro;
           bbox(bbox_id,1)= bbox(bbox_id,1)-0.5*bbox(bbox_id,3); %ת��Ϊ���ϽǶ���
           bbox(bbox_id,2)= bbox(bbox_id,2)-0.5*bbox(bbox_id,4); %ת��Ϊ���ϽǶ���                         
           bbox_id=bbox_id+1;
       end       
   end     
end
for i=1:size(bbox,1)
  %����˵���������Ͻǵ�x�����Ͻǵ�y ,��w , �� h��
  rectangle('Position',[bbox(i,1)*img_w/448,bbox(i,2)*img_h/448,bbox(i,3)*img_w/448,bbox(i,4)*img_h/448],'EdgeColor','r','LineWidth',2);  
end






