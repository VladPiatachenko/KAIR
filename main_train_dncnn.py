import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model

def main(json_path='options/train_dncnn.json', checkpoint=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('-checkpoint', type=str, default=checkpoint, help='Checkpoint file to start training from.')
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    if args.checkpoint:
        checkpoint_name = args.checkpoint.split('.')[0]
        load_path_G = os.path.join(opt['path']['models'], f"{checkpoint_name}_g.pth")
        current_step = int(''.join(filter(str.isdigit, checkpoint_name))) + 1
    else:
        # Update opt
        init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
        opt['path']['pretrained_netG'] = init_path_G
        current_step = init_iter

    option.save(opt)

    opt = option.dict_to_nonedict(opt)

    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset_type = opt['datasets']['train']['dataset_type']
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    model = define_Model(opt)

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):
            current_step += 1

            if dataset_type == 'dnpatch' and current_step % 20000 == 0:  # for 'train400'
                train_loader.dataset.update_data()

            model.update_learning_rate(current_step)
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            if current_step % opt['train']['checkpoint_test'] == 0:
                avg_psnr = 0.0
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    model.feed_data(test_data)
                    model.test()
                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])
                    current_psnr = util.calculate_psnr(E_img, H_img, border=0)
                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, os.path.basename(test_data['L_path'][0]), current_psnr))
                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')

if __name__ == '__main__':
    main()
