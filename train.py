import torch
import os
import time
from torch.utils.data import DataLoader
from JAMSNet.model.JAMSNet import JAMSNet
from JAMSNet.Dataset.LoadDataset import LoadDataset
from JAMSNet.loss.loss import cal_negative_pearson


def train(model_to_train, num_batch, dataset_loader, optimizer):
    start = time.time()
    running_loss = 0.0
    cnt = 1
    total_loss = 0.0
    num_batch_show = 500
    for i, batch_data in enumerate(dataset_loader):

        stRep_0 = batch_data[0]
        stRep_1 = batch_data[1]
        stRep_2 = batch_data[2]
        gtPPG = batch_data[3]

        gtPPG = gtPPG.squeeze(0)       # 150 1
        stRep_0 = stRep_0.squeeze(0).permute(3, 2, 0, 1)   # 150 3 192 128
        stRep_1 = stRep_1.squeeze(0).permute(3, 2, 0, 1)   # 150 3 96 64
        stRep_2 = stRep_2.squeeze(0).permute(3, 2, 0, 1)   # 150 3 48 32

        stRep_0 = stRep_0.to(device)
        stRep_1 = stRep_1.to(device)
        stRep_2 = stRep_2.to(device)
        gtPPG = gtPPG.to(device)

        stRep_pred_L = model_to_train(stRep_0, stRep_1, stRep_2)
        loss = cal_negative_pearson(stRep_pred_L, gtPPG)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print status
        running_loss += loss.item()
        total_loss += loss.item()
        cnt = cnt + 1

        if i % num_batch_show == num_batch_show - 1:
            print('[Epoch: %d, Sample: %3d/%3d] loss: %.3f' %
                  (epoch + 1, i + 1, num_batch, running_loss / cnt))
            running_loss = 0.0
            cnt = 0.0
        if i == num_batch - 1:     #
            print('[Epoch: %d, Sample: %3d/%3d] loss: %.3f' %
                  (epoch + 1, i + 1, num_batch, running_loss / cnt))
            running_loss = 0.0
            cnt = 0.0

    end = time.time()    # 训练结束时间

    return total_loss / num_batch, (end - start) / 60  # 平均损失函数值，训练所用时间


def validate(model_to_val, num_batch, dataset_loader):
    start = time.time()
    total_loss = 0.0
    for i, batch_data in enumerate(dataset_loader):

        stRep_0 = batch_data[0]
        stRep_1 = batch_data[1]
        stRep_2 = batch_data[2]
        gtPPG = batch_data[3]

        gtPPG = gtPPG.squeeze(0)
        stRep_0 = stRep_0.squeeze(0).permute(3, 2, 0, 1)
        stRep_1 = stRep_1.squeeze(0).permute(3, 2, 0, 1)
        stRep_2 = stRep_2.squeeze(0).permute(3, 2, 0, 1)

        stRep_0 = stRep_0.to(device)
        stRep_1 = stRep_1.to(device)
        stRep_2 = stRep_2.to(device)
        gtPPG = gtPPG.to(device)

        with torch.no_grad():
            stRep_pred_L = model_to_val(stRep_0, stRep_1, stRep_2)
        loss = cal_negative_pearson(stRep_pred_L, gtPPG)
        total_loss += loss.item()

    end = time.time()

    return total_loss / num_batch, (end - start) / 60


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nChn = 3
    winLength = 150
    nEpochs = 8
    workers = 4
    cnt = 0
    running_loss = 0
    dataset = 'PURE_pyramid_data'
    modelName = 'JAMSNet'
    learning_rate = 1e-4
    resume_epoch = 0

    model = JAMSNet()
    model = model.to(device)

    data_folder = 'E:\\W.H.S\\Heart_rate\\%s_%d\\json' % (dataset, winLength)  # Addresses of training set, validation set and test set

    train_dataset = LoadDataset(data_folder, split='train')
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=workers)

    val_dataset = LoadDataset(data_folder, split='val')
    val_loader = DataLoader(val_dataset, shuffle=True, num_workers=workers)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, weight_decay=0.000)

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))

    num_batch_train = len(train_dataset)
    num_batch_val = len(val_dataset)
    for epoch in range(resume_epoch, nEpochs):

        print("Training model {} epoch {} ...".format(modelName, epoch+1))
        trn_loss, trn_time = train(model, num_batch_train, train_loader, optimizer)
        print("train mean loss:", trn_loss)

        print("Valing model {} epoch {} ...".format(modelName, epoch+1))
        val_loss, val_time = validate(model, num_batch_val, val_loader)
        print("val mean loss:", val_loss)
        print("Training time: %.2f min , validation time: %.2f min" % (trn_time, val_time))

        # Save checkpoint
        save_path = '.\\weight_%s_%d\\' % (dataset, winLength)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        modelDir = save_path + '%s_%s_%d_epoch%02d.pth' % (modelName, dataset, winLength, epoch)

        state = {'epoch': epoch,
                 'train_loss': trn_loss,
                 'val_loss': val_loss,
                 'trn_time': trn_time,
                 'val_time': val_time,
                 'model': model}

        torch.save(state, modelDir)
