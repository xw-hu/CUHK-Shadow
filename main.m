%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CUHKShadow Evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This code is used for evaluating shadow detection results on CUHKShadow.
% Feel free to use the code. Please cite the following papers when you use it
%    [1] Xiaowei Hu, Tianyu Wang, Chi-Wing Fu, Yitong Jiang, Qiong Wang, and Pheng-Ann Heng. 
%    "Revisiting Shadow Detection: A New Benchmark Dataset for Complex World." IEEE TIP, 2021.
%
% (1) BER: ComputeBERonSet.m
% This code is used for computing Balanced Error Rate (BER)
% Feel free to use the code. Please cite the following papers when you use it
%    [2] Tomas F. Yago Vicente, Minh Hoai, Dimitris Samaras, "Noisy Label Recovery for Shadow Detection in Unfamiliar Domains." CVPR 2016
%    [3] Tomas F. Yago Vicente, Le Hou, Chen-Ping Yu, Minh Hoai, and Dimitris Samaras, "Large-scale training of shadow detectors with noisily-annotated shadow examples." ECCV 2016
%    [4] Vu Nguyen, Tomas F. Yago Vicente, Maozheng Zhao, Minh Hoai, Dimitris Samaras, "Shadow Detection with Conditional Generative Adversarial Networks." ICCV 2017
%
% (2) WFb: weighted_F.m
% This code is used for comuting the Weighted F-beta measure 
% Feel free to use the code. Please cite this paper when you use it
%    [5] Margolin et. al. "How to Evaluate Foreground Maps?" CVPR 2014
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


image_root = {'shadow_ADE', 'shadow_KITTI', 'shadow_MAP', 'shadow_USR', 'shadow_WEB'};
mask_root = {'mask_ADE', 'mask_KITTI', 'mask_MAP', 'mask_USR', 'mask_WEB'};

type = 'val';

mask_path = '../../CUHKshadow/';
result_path = '/home/xwhu/PycharmProjects/FSDNet/ckpt/FSDNet/(FSDNet) CUHKShadow_prediction_50000/';

acc_final = zeros(length(mask_root));
final_BER = zeros(length(mask_root));
pErr = zeros(length(mask_root));
nErr = zeros(length(mask_root));

for i=1:length(mask_root)
    
    [acc_final(i), final_BER(i), pErr(i), nErr(i), stats{i}] = ComputeBERonSet([mask_path type '/' mask_root{i} '/' ],[result_path type '/' image_root{i} '/' ]);
    
    %%%%% weighted F-measure
    weighted_F_score{i} = weighted_F_dataset([mask_path type '/' mask_root{i} '/' ],[result_path type '/' image_root{i} '/' ]);
  
    
    fprintf('%s-- wFb: %.2f, BER: %.2f, pErr: %.2f, nErr: %.2f, acc: %.4f\n', image_root{i}, mean2(weighted_F_score{i})*100, final_BER(i), pErr(i), nErr(i), acc_final(i));
end

final_stats = [stats{1}; stats{2}; stats{3}; stats{4}; stats{5}];
posAcc = sum(final_stats(:,1)) / sum(final_stats(:,3));
negAcc = sum(final_stats(:,2)) / sum(final_stats(:,4));
BER = 0.5 * (2 - posAcc - negAcc);

total_accuary = (sum(final_stats(:,1)) + sum(final_stats(:,2))) / (sum(final_stats(:,3)) + sum(final_stats(:,4)));
total_BER = 100*BER;
total_pErr = 100*(1-posAcc);
total_nErr = 100*(1-negAcc);

final_weighted_F_score = [weighted_F_score{1}, weighted_F_score{2}, weighted_F_score{3}, weighted_F_score{4}, weighted_F_score{5}];

fprintf('%s overall-- wFb: %.2f, BER: %.2f, pErr: %.2f, nErr: %.2f, acc: %.4f\n', type, mean2(final_weighted_F_score)*100, total_BER, total_pErr, total_nErr, total_accuary);



