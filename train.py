import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from data.data import SSTDataset, ToTensor
from model.net import SST_Inpainting_Model
from options.train_options import TrainOptions
import warnings

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# 加载训练参数
config = TrainOptions().parse()

# 加载数据集
print('loading data..')
dataset = SSTDataset(r'../data/ECMWF SST daily 0.05/train', transform=transforms.Compose([ToTensor()]))
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
print('data loaded..')

# 配置模型
print('configuring model..')
ourModel = SST_Inpainting_Model(in_ch=3, out_ch=3, opt=config)
ourModel.print_networks()
print('model configured..')
print('training starting..')
writer = SummaryWriter(log_dir=config.model_folder)
cnt = 0
for epoch in range(config.epochs):

    for i, data in enumerate(dataloader):
        gt = data['gt'].to(device)
        # print(gt.shape)
        # normalize to values between -1 and 1
        gt = gt / 127.5 - 1

        data_in = {'gt': gt}
        ourModel.setInput(data_in)
        ourModel.optimizer_parameters()

        if (i + 1) % config.viz_steps == 0:
            ret_loss = ourModel.get_current_losses()
            print('[%d, %5d]' % (epoch + 1, i + 1))
            print('G_loss: %.5f (rec: %.5f, ae: %.5f, adv: %.5f, mrf: %.5f), D_loss: %.5f' %
                  (ret_loss['G_loss'], ret_loss['G_loss_rec'], ret_loss['G_loss_ae'],
                   ret_loss['G_loss_adv'], ret_loss['G_loss_mrf'], ret_loss['D_loss']))
            print('H_loss: %.5f (rec: %.5f, ae: %.5f, adv: %.5f, mrf: %.5f), D_H_loss: %.5f' %
                  (ret_loss['H_loss'], ret_loss['H_loss_rec'], ret_loss['H_loss_ae'],
                   ret_loss['H_loss_adv'], ret_loss['H_loss_mrf'], ret_loss['D_H_loss']))
            writer.add_scalar('G_adv_loss', ret_loss['G_loss_adv'], cnt)
            writer.add_scalar('H_adv_loss', ret_loss['H_loss_adv'], cnt)
            writer.add_scalar('D_loss', ret_loss['D_loss'], cnt)
            writer.add_scalar('D_H_loss', ret_loss['D_H_loss'], cnt)
            writer.add_scalar('G_mrf_loss', ret_loss['G_loss_mrf'], cnt)
            writer.add_scalar('H_mrf_loss', ret_loss['H_loss_mrf'], cnt)
            images = ourModel.get_current_visuals_tensor()
            G_completed = vutils.make_grid(images['G_completed'], normalize=True, scale_each=True)
            H_completed = vutils.make_grid(images['H_completed'], normalize=True, scale_each=True)
            im_input = vutils.make_grid(images['input'], normalize=True, scale_each=True)
            im_gt = vutils.make_grid(images['gt'], normalize=True, scale_each=True)
            writer.add_scalar('G_loss', ret_loss['G_loss'], cnt)
            writer.add_scalar('H_loss', ret_loss['H_loss'], cnt)
            writer.add_scalar('G_reconstruction_loss', ret_loss['G_loss_rec'], cnt)
            writer.add_scalar('H_reconstruction_loss', ret_loss['G_loss_rec'], cnt)
            writer.add_scalar('G_autoencoder_loss', ret_loss['G_loss_ae'], cnt)
            writer.add_scalar('H_autoencoder_loss', ret_loss['H_loss_ae'], cnt)
            if (i + 1) % config.train_spe == 0:
                print('saving model ..')
                ourModel.save_networks(epoch + 1)
            cnt += 1
        # if i == 0:
        break
    break
    ourModel.save_networks(epoch + 1)
writer.export_scalars_to_json(os.path.join(config.model_folder, 'MSGC_scalars.json'))
writer.close()
