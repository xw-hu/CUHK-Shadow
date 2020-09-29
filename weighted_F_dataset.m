function [weighted_F_score] = weighted_F_dataset(GTFolder, PredFolder)




dImg = dir([GTFolder,'*.*']);
rImg = dir([PredFolder,'*.*']);

dImg(1)=[];
dImg(1)=[];
rImg(1)=[];
rImg(1)=[];

nImg = numel(dImg);
weighted_F_score = zeros(1,nImg);
for i = 1 : nImg
    
    GT = imread([GTFolder,dImg(i).name]);
	prediction = imread([PredFolder,rImg(i).name]);
    
%     if sum(sum(GT(:,:))) == 0
%         GT(1,1)=255;
%     end
    if numel(size(GT))>2
        GT = rgb2gray(GT);
    end
    GT = logical(GT);
    
    if numel(size(prediction))>2
        prediction = rgb2gray(prediction);
    end

    % Normalize the prediction.
    d_prediction = double(prediction);

    d_prediction = d_prediction./255;


    Q = weighted_F(d_prediction,GT);
    weighted_F_score(i) = Q;
end

