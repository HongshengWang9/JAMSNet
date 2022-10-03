import torch
import os
import numpy as np
import json
import scipy.io as io
from scipy.io import loadmat
import time

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    workers = 4
    num_epoches = 8
    winLength = 150
    N = 512
    fps = 30
    kers = 3
    num_batch_show = 500
    dataset_test = 'PURE_pyramid_data'
    modelName = 'JAMSNet'
    data_folder = 'E:\\W.H.S\\Heart_rate\\%s_%d\\json' % (dataset_test, winLength)
    test_image_json_file = data_folder + '\\test_image.json'
    test_label_json_file = data_folder + '\\test_label.json'
    with open(test_image_json_file, 'r') as j:
        test_image_list = json.load(j)
    with open(test_label_json_file, 'r') as j:
        test_label_list = json.load(j)

    num_test_file = len(test_image_list)
    print("testing " + data_folder)
    list_epoch = []
    list_snr_loss = []
    list_mae_loss = []
    list_rmse_loss = []
    for i_epoch in range(num_epoches):
        modelDir = '.\\weight_%s_%d\\%s_%s_%d_epoch%02d.pth' \
                 % (dataset_test, winLength, modelName, dataset_test, winLength, i_epoch)
        checkpoint = torch.load(modelDir)
        model = checkpoint['model']
        model = model.to(device)

        for i in range(num_test_file):

            if i % num_batch_show == num_batch_show - 1:
                print("Testing the %d/%d sample, epoch %02d..." % (i + 1, num_test_file, i_epoch))

            if i == num_test_file - 1:
                print("Testing the %d/%d sample, epoch %02d..." % (i + 1, num_test_file, i_epoch))

            file_name_image = test_image_list[i]
            stRep = loadmat(file_name_image)
            stRep_0 = stRep['layer0']
            stRep_1 = stRep['layer1']
            stRep_2 = stRep['layer2']
            stRep_0 = stRep_0 / 255
            stRep_1 = stRep_1 / 255
            stRep_2 = stRep_2 / 255
            stRep_0 = torch.FloatTensor(stRep_0).permute(3, 2, 0, 1)
            stRep_1 = torch.FloatTensor(stRep_1).permute(3, 2, 0, 1)
            stRep_2 = torch.FloatTensor(stRep_2).permute(3, 2, 0, 1)
            stRep_0 = stRep_0.to(device)
            stRep_1 = stRep_1.to(device)
            stRep_2 = stRep_2.to(device)
            ppg_est_L = model(stRep_0, stRep_1, stRep_2)
            ppg_est = ppg_est_L.cpu()
            ppg_est = ppg_est.detach().numpy()

            file_path = file_name_image.split('\\')
            save_path = '.\\test_%s_%d\\%s\\epoch%02d\\%s' \
                        % (dataset_test, winLength, file_path[3], i_epoch, file_path[4])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            ppg_est_savepath = save_path + '\\ppg_est' + '%s' % file_path[5]
            io.savemat(ppg_est_savepath, {'ppg_est': ppg_est})

