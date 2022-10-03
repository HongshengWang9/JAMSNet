clear;
addpath('.\utils');

winLength = 150;
destFold_image =  'E:\W.H.S\Heart_rate\UBFC_pyramid_data_150';
destFold_lable =  'E:\W.H.S\Heart_rate\UBFC_pyramid_data_150';

save_file_name = 'UBFC_train_val_test_idx_list_1.mat';
load(save_file_name); % sub_idx_train, sub_idx_val, sub_idx_test
nSub_train = length(sub_idx_train);
nSub_val = length(sub_idx_val);
nSub_test = length(sub_idx_test);

savejson = ['E:\W.H.S\Heart_rate\UBFC_pyramid_data_' num2str(winLength) '\json_1'];
if ~exist(savejson,'dir') 
      mkdir(savejson);
end

% write train json file
saveFile_image = [savejson '\train_image.json'];
saveFile_label = [savejson '\train_label.json'];

jsonText_image = {};
jsonText_label = {};
jsonText_cnt = 1;
for subIdx = 1:nSub_train
    iSub = sub_idx_train(subIdx);   
    subID = ['subject' num2str(iSub,'%d')];
    vidDir = [destFold_lable '\' subID];
    if ~exist(vidDir,'file')
        continue;
    end
    nMat = 0.5*(length(dir(vidDir))-2);
    for iMat = 1:nMat
        imageFile = [destFold_image '\' subID '\' num2str(iMat,'%02d') '.mat'];
        labelFile = [destFold_lable '\' subID '\gtPPG' num2str(iMat,'%02d') '.mat'];
        jsonText_image{jsonText_cnt} = imageFile;
        jsonText_label{jsonText_cnt} = labelFile;
        jsonText_cnt = jsonText_cnt +1;
    end
end    
writeJSONfile(saveFile_image,jsonText_image);
writeJSONfile(saveFile_label,jsonText_label);

% write val json file
saveFile_image = [savejson '\val_image.json'];
saveFile_label = [savejson '\val_label.json'];
jsonText_image = {};
jsonText_label = {};
jsonText_cnt = 1;
for subIdx = 1:nSub_val
    iSub = sub_idx_val(subIdx);   
    subID = ['subject' num2str(iSub,'%d')];
    vidDir = [destFold_lable '\' subID];
    if ~exist(vidDir,'file')
        continue;
    end
    nMat = 0.5*(length(dir(vidDir))-2);
    for iMat = 1:nMat
        imageFile = [destFold_image '\' subID '\' num2str(iMat,'%02d') '.mat'];
        labelFile = [destFold_lable '\' subID '\gtPPG' num2str(iMat,'%02d') '.mat'];
        jsonText_image{jsonText_cnt} = imageFile;
        jsonText_label{jsonText_cnt} = labelFile;
        jsonText_cnt = jsonText_cnt +1;
    end
end    
writeJSONfile(saveFile_image,jsonText_image);
writeJSONfile(saveFile_label,jsonText_label);

% write test json file
saveFile_image = [savejson '\test_image.json'];
saveFile_label = [savejson '\test_label.json'];
jsonText_image = {};
jsonText_label = {};
jsonText_cnt = 1;
for subIdx = 1:nSub_test
    iSub = sub_idx_test(subIdx);   
    subID = ['subject' num2str(iSub,'%d')];
    vidDir = [destFold_lable '\' subID];
    if ~exist(vidDir,'file')
        continue;
    end
    nMat = 0.5*(length(dir(vidDir))-2);
    for iMat = 1:nMat
        imageFile = [destFold_image '\' subID '\' num2str(iMat,'%02d') '.mat'];
        labelFile = [destFold_lable '\' subID '\gtPPG' num2str(iMat,'%02d') '.mat'];
        jsonText_image{jsonText_cnt} = imageFile;
        jsonText_label{jsonText_cnt} = labelFile;
        jsonText_cnt = jsonText_cnt +1;
    end
end    
writeJSONfile(saveFile_image,jsonText_image);
writeJSONfile(saveFile_label,jsonText_label);
