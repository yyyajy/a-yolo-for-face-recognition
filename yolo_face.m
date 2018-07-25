clear;
addpath('..') %加入+caffe路径  
addpath('.')
caffe.set_mode_cpu();%设置CPU模式  
model = 'D:/CelebA/matlab_demo/yolo/yolo/yolo-face-deployori.prototxt';%模型deploy  
weights = 'D:/CelebA/matlab_demo/yolo/yolo/yolo-face.caffemodel';%参数  .
%  weights = 'D:/CelebA/matlab_demo/yolo/yolo/_iter_65000.caffemodel';
net=caffe.Net(model,weights,'test');%测试  
pdollar_toolbox_path='C:\caffe-master\matlab\demo\yolo';
addpath(genpath(pdollar_toolbox_path));
side=11;
B=2;
class_index=1;%类别的索引
confidence_index=side*side+1;%置信度的索引
loc_index=confidence_index+side*side*B;
thresh = 0.25;%设定阈值，当结果大于这个阈值时，将box画出
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
feat=net.blobs('fc12').get_data();%以获取到1331个数值
%下面对这1331个数值做处理
for ii=1:side*side%121个位置  先行后列
   row=floor((ii-1)/side+1);
   col=ii-side*(row-1);
   class_index=ii;%每个网格的类别概率索引
   for jj=1:B% 每个位置上的B个框
       confidence_index=side*side+B*(ii-1)+jj; %每个box的置信度索引     
       pro=feat(class_index,1)*feat(confidence_index,1); % 类别信息*置信度    
       if pro>thresh           
           loc_index=side*side+side*side*2+1+((ii-1)*B+jj-1)*4;%四个坐标值的起始位置x
           bbox(bbox_id,1) = (feat(loc_index,1)+col-1)*448/side;%X box的中心位置
           bbox(bbox_id,2) = (feat(loc_index+1,1)+row-1)*448/side;%y box的中心位置
           bbox(bbox_id,3) = (feat(loc_index+2,1)*feat(loc_index+2,1))*448;%w
           bbox(bbox_id,4) = (feat(loc_index+3,1)*feat(loc_index+3,1))*448;%h 
           bbox(bbox_id,5)=pro;
           bbox(bbox_id,1)= bbox(bbox_id,1)-0.5*bbox(bbox_id,3); %转换为左上角顶点
           bbox(bbox_id,2)= bbox(bbox_id,2)-0.5*bbox(bbox_id,4); %转换为左上角顶点                         
           bbox_id=bbox_id+1;
       end       
   end     
end
for i=1:size(bbox,1)
  %参数说明：（左上角点x，左上角点y ,宽w , 高 h）
  rectangle('Position',[bbox(i,1)*img_w/448,bbox(i,2)*img_h/448,bbox(i,3)*img_w/448,bbox(i,4)*img_h/448],'EdgeColor','r','LineWidth',2);  
end






