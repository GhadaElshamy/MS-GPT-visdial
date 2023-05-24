import gc
import os
import json
import options
import pprint
import random
import numpy as np
from time import gmtime, strftime
from timeit import default_timer as timer

# XLA imports
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

import torch
import torchvision
import torch.nn as nn

from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset

import torch.optim as optim
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
from pytorch_transformers.optimization import AdamW

from dataloader.dataloader_visdial_gen import VisdialDataset
from dataloader.dataloader_cc12m_gen import CC12mDataset
from models.visual_dialog_decoder_gpt import GPTVisualDialogDecoder
from models.visual_dialog_encoder import VisualDialogEncoder
from models.visual_dialog_decoder import VisualDialogDecoder
from models.visual_dialog_model import EncoderDecoderModel

from utils.data_utils import sequence_mask, batch_iter
from utils.logger import Logger
from utils.optim_utils import WarmupLinearScheduleNonZero


# configurations
os.environ['XLA_USE_BF16'] = '1'
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '1000000000'

#TODO: For calculating loss over the tpu cores
class AverageMeter(object):
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def forward(model, batch, params):
    enc_next_sentence_labels = None
    enc_image_target = None
    enc_image_label = None
    dec_labels = None

    # language stuff
    enc_input_ids = batch['enc_input_ids']
    enc_segments = batch['enc_segments']
    enc_sep_indices = batch['enc_sep_indices']
    enc_mlm_labels = batch['enc_mlm_labels']
    enc_hist_len = batch['enc_hist_len']
    enc_att_mask = batch['enc_att_mask']
    dec_input_ids = batch['dec_input_ids']
    dec_att_mask = batch['dec_att_mask']

    enc_input_ids = enc_input_ids.view(-1, enc_input_ids.shape[-1])
    enc_segments = enc_segments.view(-1, enc_segments.shape[-1])
    enc_sep_indices = enc_sep_indices.view(-1, enc_sep_indices.shape[-1])
    enc_mlm_labels = enc_mlm_labels.view(-1, enc_mlm_labels.shape[-1])
    enc_hist_len = enc_hist_len.view(-1)
    enc_att_mask = enc_att_mask.view(-1, enc_att_mask.shape[-1])
    dec_input_ids = dec_input_ids.view(-1, dec_input_ids.shape[-1])
    dec_att_mask = dec_att_mask.view(-1, dec_att_mask.shape[-1])

    # image stuff
    orig_features = batch['enc_image_feat']
    orig_spatials = batch['enc_image_loc']
    orig_image_mask = batch['enc_image_mask']

    enc_image_features = orig_features.view(-1, orig_features.shape[-2], orig_features.shape[-1])
    enc_image_spatials = orig_spatials.view(-1, orig_spatials.shape[-2], orig_spatials.shape[-1])
    enc_image_mask = orig_image_mask.view(-1, orig_image_mask.shape[-1])

    if 'train' in params['mode']:
        # random sampling of valid data
        dec_labels = batch['dec_labels']
        dec_labels = dec_labels.view(-1, dec_labels.shape[-1])
        cand_samples = (dec_labels.sum(-1) != 0).float()
        sample_indices = torch.multinomial(cand_samples, params['batch_size'], replacement=True)
    else:
        sample_indices = torch.arange(enc_hist_len.shape[0])

    enc_input_ids = enc_input_ids[sample_indices, :]
    enc_segments = enc_segments[sample_indices, :]
    enc_sep_indices = enc_sep_indices[sample_indices, :]
    enc_mlm_labels = enc_mlm_labels[sample_indices, :]
    enc_hist_len = enc_hist_len[sample_indices]
    enc_att_mask = enc_att_mask[sample_indices, :]
    dec_input_ids = dec_input_ids[sample_indices, :]
    dec_att_mask = dec_att_mask[sample_indices, :]
    enc_image_features = enc_image_features[sample_indices, :, :]
    enc_image_spatials = enc_image_spatials[sample_indices, :, :]
    enc_image_mask = enc_image_mask[sample_indices, :]

    if 'train' in params['mode']:
        dec_labels = dec_labels[sample_indices, :]
        dec_labels = dec_labels.to(params['device'])

        enc_next_sentence_labels = batch['enc_next_sentence_labels']
        enc_next_sentence_labels = enc_next_sentence_labels.view(-1)
        enc_next_sentence_labels = enc_next_sentence_labels[sample_indices]
        enc_next_sentence_labels = enc_next_sentence_labels.to(params['device'])

        orig_image_target = batch['enc_image_target']
        orig_image_label = batch['enc_image_label']

        enc_image_target = orig_image_target.view(-1, orig_image_target.shape[-2], orig_image_target.shape[-1])
        enc_image_label = orig_image_label.view(-1, orig_image_label.shape[-1])

        enc_image_target = enc_image_target[sample_indices, :, :]
        enc_image_label = enc_image_label[sample_indices, :]

        enc_image_target = enc_image_target.to(params['device'])
        enc_image_label = enc_image_label.to(params['device'])

    enc_input_ids = enc_input_ids.to(params['device'])
    enc_segments = enc_segments.to(params['device'])
    enc_sep_indices = enc_sep_indices.to(params['device'])
    enc_mlm_labels = enc_mlm_labels.to(params['device'])
    enc_hist_len = enc_hist_len.to(params['device'])
    enc_att_mask = enc_att_mask.to(params['device'])
    dec_input_ids = dec_input_ids.to(params['device'])
    dec_att_mask = dec_att_mask.to(params['device'])

    enc_image_features = enc_image_features.to(params['device'])
    enc_image_spatials = enc_image_spatials.to(params['device'])
    enc_image_mask = enc_image_mask.to(params['device'])

    lm_loss, lm_scores = model(
        enc_image_features=enc_image_features,
        enc_image_spatials=enc_image_spatials,
        enc_image_mask=enc_image_mask,
        enc_image_target=enc_image_target,
        enc_image_label=enc_image_label,
        enc_next_sentence_labels=enc_next_sentence_labels,
        enc_input_ids=enc_input_ids,
        enc_segments=enc_segments,
        enc_sep_indices=enc_sep_indices,
        enc_mlm_labels=enc_mlm_labels,
        enc_attention_mask=enc_att_mask,
        dec_input_ids=dec_input_ids,
        dec_attention_mask=dec_att_mask,
        dec_labels=dec_labels
    )
    if 'train' in params['mode']:
        lm_loss = lm_loss.mean()

    return lm_loss, lm_scores

#TODO:Training TPU
def train_fn(model, epoch_id, num_iter_epoch, para_loader, criterion, optimizer, scheduler, device, params, logger, batch_verbose=100):
    # initialize
    model.train()
    trn_loss_meter = AverageMeter()


    # training loop
    for batch_idx, batch in enumerate(para_loader):

        iter_id = batch_idx + (epoch_id * num_iter_epoch)

        # expand image features,
        orig_features = batch['enc_image_feat']
        orig_spatials = batch['enc_image_loc']
        orig_image_mask = batch['enc_image_mask']
        orig_image_target = batch['enc_image_target']
        orig_image_label = batch['enc_image_label']

        num_rounds = batch["enc_input_ids"].shape[1]
        num_samples = batch["enc_input_ids"].shape[2]

        features = orig_features.unsqueeze(1).unsqueeze(1).expand(orig_features.shape[0], num_rounds, num_samples,
                                                                  orig_features.shape[1],
                                                                  orig_features.shape[2]).contiguous()
        spatials = orig_spatials.unsqueeze(1).unsqueeze(1).expand(orig_spatials.shape[0], num_rounds, num_samples,
                                                                  orig_spatials.shape[1],
                                                                  orig_spatials.shape[2]).contiguous()
        image_label = orig_image_label.unsqueeze(1).unsqueeze(1).expand(orig_image_label.shape[0], num_rounds,
                                                                        num_samples,
                                                                        orig_image_label.shape[1]).contiguous()
        image_mask = orig_image_mask.unsqueeze(1).unsqueeze(1).expand(orig_image_mask.shape[0], num_rounds, num_samples,
                                                                      orig_image_mask.shape[1]).contiguous()
        image_target = orig_image_target.unsqueeze(1).unsqueeze(1).expand(orig_image_target.shape[0], num_rounds,
                                                                          num_samples, orig_image_target.shape[1],
                                                                          orig_image_target.shape[2]).contiguous()

        batch['enc_image_feat'] = features.contiguous()
        batch['enc_image_loc'] = spatials.contiguous()
        batch['enc_image_mask'] = image_mask.contiguous()
        batch['enc_image_target'] = image_target.contiguous()
        batch['enc_image_label'] = image_label.contiguous()
        ########

        # extract inputs and labels
        # inputs = inputs.to(device)
        # labels = labels.to(device)

        # forward and backward pass
        preds = model(batch)
        # loss = criterion(preds, labels)
        loss, _ = forward(model, batch, params)
        loss.backward()

        if iter_id > 0:
            xm.optimizer_step(optimizer,
                          barrier=True)  # barrier is required on single-core training but can be dropped with multiple cores
            optimizer.zero_grad()

        # scheduler step
        scheduler.step()

        # compute loss
        trn_loss_meter.update(loss.detach().item(), batch.size(0))

        if iter_id % 10 == 0:
            end_t = timer()
            cur_lr = optimizer.param_groups[0]['lr']
            cur_epoch = float(iter_id) / num_iter_epoch
            timestamp = strftime('%a %d %b %y %X', gmtime())
            print_lm_loss = loss.item()

            print_format = '[%s][LR: %.7f][Ep: %.2f][Iter: %d][Time: %5.2fs][LM Loss: %.4g]'
            print_info = [
                timestamp, cur_lr, cur_epoch, iter_id, end_t - start_t, print_lm_loss
            ]
            logger.write(print_format % tuple(print_info))
            start_t = end_t

        # feedback
        if (batch_idx > 0) and (batch_idx % batch_verbose == 0):
            xm.master_print('-- batch {} | cur_loss = {:.6f}, avg_loss = {:.6f}'.format(
                batch_idx, loss.item(), trn_loss_meter.avg))

        # clear memory
        del batch, preds, loss
        gc.collect()

        # early stop
        # if batch_idx > batches_per_epoch:
        #     break



    # clear memory
    del para_loader, batch_idx
    gc.collect()

    return trn_loss_meter.avg


def _run(params):
    ### DATA PREP

    # select mode (train vd or cc12m)
    mode = params['mode']
    assert mode == 'vd_train' or mode == 'cc12m_train'
    assert params['model'] == 'enc_dec_a' or params['model'] == 'enc_dec_q'

    # logger init
    logger = Logger(os.path.join(params['save_path'], 'log_%s.txt' % mode))
    logger.write(str(params))

    # prepare dataset (visDial or cc12m)
    if mode == 'vd_train':
        datasets = VisdialDataset(params)
        datasets.mode = 'vd_train'
        num_iter_epoch = datasets.numDataPoints[mode] // params['batch_size']
    else:
        datasets = []
        total_datapoints = 0
        image_feat_path = params['cc12m_image_feats']
        dialog_path = params['cc12m_processed_train']

        for n in range(params['iter']):
            iter_path = dialog_path + 'iter%s/' % (n + 1)
            data_list = [x for x in range(int(params['chunk']))]
            for i in data_list:
                params['cc12m_image_feats'] = image_feat_path + "cc12m_img_feat_%d.lmdb" % i
                params['cc12m_processed_train'] = iter_path + "cc12m_dialogs_%d.txt" % i
                dataset = CC12mDataset(params)
                dataset.mode = 'cc12m_train'
                datasets.append(dataset)
                total_datapoints += dataset.numDataPoints
            print('iteration %d data loaded' % (n + 1))
        num_iter_epoch = total_datapoints // params['batch_size']
        datasets = ConcatDataset(datasets)
    step_total = num_iter_epoch * 100
    warmup_steps = 1500

    device = xm.xla_device()
    params['device'] = device
    dialog_encoder = VisualDialogEncoder(params)
    dialog_decoder = None
    # TODO Choose the required decoder
    if params['dec_model'] == 'gpt2':
        dialog_decoder = GPTVisualDialogDecoder(params)
    else:
        dialog_decoder = VisualDialogDecoder(params)

    model = EncoderDecoderModel(params, dialog_encoder, dialog_decoder)
    mx = xmp.MpModelWrapper(model)

    # data samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(datasets,
                                                                    num_replicas=xm.xrt_world_size(),
                                                                    rank=xm.get_ordinal(),
                                                                    shuffle=True)

    train_loader = torch.utils.data.DataLoader(datasets,
                                               batch_size=params['batch_size'],
                                               sampler=train_sampler,
                                               num_workers=0,
                                               pin_memory=True)

    ### MODEL PREP

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    langauge_weights = None
    with open('config/language_weights.json') as f:
        langauge_weights = json.load(f)

    optimizer_grouped_parameters = []
    for key, value in dict(dialog_encoder.named_parameters()).items():
        if value.requires_grad:
            if key in langauge_weights:
                lr = params['lr']
            else:
                lr = params['image_lr']

            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0}
                ]

            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]

    for key, value in dict(dialog_decoder.named_parameters()).items():
        if value.requires_grad:
            if key in langauge_weights:
                lr = params['lr']
            else:
                lr = params['image_lr']

            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0}
                ]

            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]

    logger.write('\n%d iter per epoch.' % num_iter_epoch)
    logger.write('%d total step.' % step_total)


    # send to TPU
    device = xm.xla_device()
    model = mx.to(device)

    # scale LR
    scaled_eta = params['lr'] * xm.xrt_world_size()

    # optimizer and loss
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=scaled_eta)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    optimizer = AdamW(optimizer_grouped_parameters, lr=params['lr'])
    scheduler = WarmupLinearScheduleNonZero(optimizer, warmup_steps=warmup_steps, t_total=step_total)

    start_iter_id = 0
    start_epoch_id = 0
    ### MODELING

    if params['start_path']:
        pretrained_dict = torch.load(params['start_path'], map_location=device)
        if params['continue_training']:
            model_dict = model.state_dict()
            optimizer_dict = optimizer.state_dict()
            pretrained_dict_model = pretrained_dict['model_state_dict']
            pretrained_dict_optimizer = pretrained_dict['optimizer_state_dict']
            pretrained_dict_scheduler = pretrained_dict['scheduler_state_dict']
            pretrained_dict_model = {k: v for k, v in pretrained_dict_model.items() if k in model_dict}
            pretrained_dict_optimizer = {k: v for k, v in pretrained_dict_optimizer.items() if k in optimizer_dict}
            model_dict.update(pretrained_dict_model)
            optimizer_dict.update(pretrained_dict_optimizer)
            model.load_state_dict(model_dict)
            optimizer.load_state_dict(optimizer_dict)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            if mode in params['start_path']:
                # load the scheduler when start checkpoint and mode are the same
                scheduler = WarmupLinearScheduleNonZero(optimizer, warmup_steps=warmup_steps, \
                                                        t_total=step_total, last_epoch=pretrained_dict["iter_id"])
                scheduler.load_state_dict(pretrained_dict_scheduler)
                start_iter_id = pretrained_dict['iter_id']
                start_epoch_id = start_iter_id // num_iter_epoch
            del pretrained_dict, model_dict, optimizer_dict, pretrained_dict_model, pretrained_dict_optimizer, pretrained_dict_scheduler
            # with torch.cuda.device("cuda:%s" % params["gpu_ids"][0]):
            #     torch.cuda.empty_cache()
            gc.collect()
        else:
            if 'model_state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['model_state_dict']

            model_dict = dialog_encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            print("number of keys transferred", len(pretrained_dict))
            assert len(pretrained_dict.keys()) > 0
            model_dict.update(pretrained_dict)
            dialog_encoder.load_state_dict(model_dict)
            del pretrained_dict, model_dict


    # placeholders
    trn_losses = []
    # val_losses = []
    best_val_loss = 1

    # modeling loop
    gc.collect()
    for epoch in range(params['num_epochs']):

        # display info
        xm.master_print('-' * 55)
        xm.master_print('EPOCH {}/{}'.format(epoch + 1, params['num_epochs']))
        xm.master_print('-' * 55)
        xm.master_print('- initialization | TPU cores = {}, lr = {:.6f}'.format(
            xm.xrt_world_size(), scheduler.get_lr()[len(scheduler.get_lr()) - 1] / xm.xrt_world_size()))
        epoch_start = timer()
        gc.collect()

        # update train_loader shuffling
        train_loader.sampler.set_epoch(epoch)

        # training pass
        train_start = timer()
        xm.master_print('- training...')
        para_loader = pl.ParallelLoader(train_loader, [device])
        trn_loss = train_fn(model,
                            epoch=epoch + 1,
                            num_iter_epoch=num_iter_epoch,
                            para_loader=para_loader.per_device_loader(device),
                            criterion=None,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            device=device,
                            params=params,
                            logger=logger)

        #Save each epoch
        torch.save(
            {
                'model_state_dict': model.module.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter_id': num_iter_epoch
            },
            os.path.join(
                params['save_path'],
                '%s_%s_%d.ckpt' % (mode, params['chunk'], epoch)
            )
        )
        logger.write('\n%d epoch ended.' % epoch)

        del para_loader
        gc.collect()

        # validation pass
        # valid_start = time.time()
        # xm.master_print('- validation...')
        # para_loader = pl.ParallelLoader(valid_loader, [device])
        # val_loss = valid_fn(epoch=epoch + 1,
        #                     para_loader=para_loader.per_device_loader(device),
        #                     criterion=criterion,
        #                     device=device)
        # del para_loader
        # gc.collect()
        #
        # # save weights
        # if val_loss < best_val_loss:
        #     xm.save(model.state_dict(), 'weights_{}.pt'.format(model_name))
        #     best_val_loss = val_loss

        # display info
        xm.master_print('- elapsed time | train = {:.2f} min, valid = {:.2f} min'.format(
            (epoch_start - train_start) / 60, (0) / 60))
        xm.master_print('- average loss | train = {:.6f}, valid = {:.6f}'.format(
            trn_loss, 0.0))
        xm.master_print('-' * 55)
        xm.master_print('')

        # save losses
        trn_losses.append(trn_loss)
        # val_losses.append(val_loss)
        # del trn_loss, val_loss
        del trn_loss
        gc.collect()

    # print results
    # xm.master_print('Best results: loss = {:.6f} (epoch {})'.format(np.min(val_losses), np.argmin(val_losses) + 1))
    xm.master_print('Best results: loss = {:.6f} (epoch {})'.format(np.min(trn_losses), np.argmin(trn_losses) + 1))

    # return trn_losses, val_losses
    return trn_losses


# wrapper function
def _mp_fn(rank, flags, params):
    torch.set_default_tensor_type('torch.FloatTensor')
    trn_losses = _run(params)
    np.save('trn_losses.npy', np.array(trn_losses))
    # np.save('val_losses.npy', np.array(val_losses))





if __name__ == '__main__':

    # get arguments
    params = options.read_command_line()
    if not os.path.exists(params['save_path']):
        os.makedirs(params['save_path'], exist_ok=True)
    pprint.pprint(params)

    # modeling
    gc.collect()
    FLAGS = {}
    xmp.spawn(_mp_fn, args=(FLAGS,params), nprocs=params['num_workers'], start_method='fork')



    for epoch_id, idx, batch in batch_iter(dataloader, params, start_epoch_id):


        # optimization and scheduleing
        if iter_id > 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        if iter_id % 10 == 0:
            end_t = timer()
            cur_lr = optimizer.param_groups[0]['lr']
            cur_epoch = float(iter_id) / num_iter_epoch
            timestamp = strftime('%a %d %b %y %X', gmtime())
            print_lm_loss = lm_loss.item()

            print_format = '[%s][LR: %.7f][Ep: %.2f][Iter: %d][Time: %5.2fs][LM Loss: %.4g]'
            print_info = [
                timestamp, cur_lr, cur_epoch, iter_id, end_t - start_t, print_lm_loss
            ]
            logger.write(print_format % tuple(print_info))
            start_t = end_t


            logger.write('\n%d epoch ended.' % epoch_id)
