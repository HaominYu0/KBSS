import os
import random
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import csv
from collections import Counter
import yaml
import shutil
from scipy.ndimage import convolve1d
import sys
import dill
import time
import warnings
import numpy as np
from random import sample
from sklearn import metrics
from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from dataset.data_finetune import CIFData
from dataset.data_unseen import CIFData_unseen
from dataset.data_finetune import collate_pool, get_train_val_test_loader
from model.kbss_net import finetune_ENDE
from dataset.graph import *
from torch_geometric.data import Data, Batch
import os
from scipy.ndimage import gaussian_filter1d
from itertools import zip_longest

import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./finetune_config/config_ft.yaml', os.path.join(model_checkpoints_folder, 'config_ft.yaml'))


def _check_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

class SPLLoss(nn.NLLLoss):
    def __init__(self, n_samples=0, beta=0, thres=0):
        super(SPLLoss, self).__init__()
        self.threshold = 0.1
        self.growing_factor = 1.03
        self.v = torch.zeros(n_samples).int()
        #import ipdb
        self.thres = thres
        self.beta = beta
        #ipdb.set_trace()

    def forward(self, input, index, weight):
        pseudo_label_class = torch.softmax(input.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label_class, dim=-1)

        #super_loss = F.cross_entropy(input, target, reduction="none")
        v = self.spl_loss(max_probs, weight)
        #self.v[index] = v
        return v #(max_probs * v).mean()
    
    def increase_threshold(self):
        #import ipdb
        #ipdb.set_trace()
        self.threshold *= self.growing_factor
        if self.threshold > self.thres:
            self.threshold = self.thres
    def spl_loss(self, super_loss, weight):
        v = torch.where(super_loss > self.threshold, super_loss > self.threshold, super_loss*torch.FloatTensor(weight).cuda()*self.beta>self.threshold)

        return v.float()



class FineTune(object):
    def __init__(self, config, root_config, target_dataset, i_num, current_time):
        self.config = config
        self.root_config = root_config
        self.i_num = i_num
        self.device = self._get_device()
        dir_name = current_time
        log_dir = os.path.join('runs_nf', dir_name)
        _check_file(log_dir)
        log_dir_num =  os.path.join(log_dir, target_dataset)
        _check_file(log_dir_num)

        self.writer = SummaryWriter(log_dir=log_dir_num)

        if self.config['task'] == 'classification':
            self.criterion = nn.NLLLoss()
        else:
            self.criterion = nn.L1Loss()
        data_loader_root = 'Dataset/'
        _check_file(data_loader_root)
        data_loader_path = data_loader_root+target_dataset
        if os.path.exists(data_loader_path):
             with open(data_loader_path+'/train.pkl','rb') as f:
                   self.train_loader = dill.load(f)
             with open(data_loader_path+'/val.pkl','rb') as f:
                   self.valid_loader = dill.load(f)
             with open(data_loader_path+'/train_unseen.pkl','rb') as f:
                  self.useen_train_loader = dill.load(f)
             with open(data_loader_path+'/test.pkl','rb') as f:
                   self.test_loader = dill.load(f)
             with open(data_loader_path+'/sample.pkl', 'rb') as f:
                   sample_target = dill.load(f)

        else:
            self.dataset_train = CIFData('train', i_num,  self.config['task'], **self.config['dataset'])
            self.dataset_unseen = CIFData_unseen('train_unseen', i_num,  self.config['task'], **self.config['dataset'])
            self.dataset_val   =  CIFData('val', i_num, self.config['task'], **self.config['dataset'])
            self.dataset_test  =  CIFData('test', i_num, self.config['task'], **self.config['dataset'])
            self.target_dataset = target_dataset
            sample_target = torch.tensor(self.dataset_train.id_goal+self.dataset_val.id_goal_val)
            self.random_seed = self.config['random_seed']
            collate_fn = collate_pool
            self.train_loader, self.useen_train_loader, self.valid_loader, self.test_loader = get_train_val_test_loader(dataset_train = self.dataset_train, dataset_unseen=  self.dataset_unseen, dataset_val = self.dataset_val,dataset_test = self.dataset_test,collate_fn = collate_fn,pin_memory = self.config['cuda'],batch_size = self.root_config['batch_size'],return_test = True,**self.root_config['dataloader'])
            _check_file(data_loader_path)
            with open(data_loader_path+'/train.pkl','wb') as f:
                  dill.dump(self.train_loader, f)
            with open(data_loader_path+'/train_unseen.pkl','wb') as f:
                  dill.dump(self.useen_train_loader, f)
            with open(data_loader_path+'/val.pkl','wb') as f:
                  dill.dump(self.valid_loader, f)
            with open(data_loader_path+'/test.pkl','wb') as f:
                  dill.dump(self.test_loader, f)
            with open(data_loader_path+'/sample.pkl','wb') as f:
                  dill.dump(sample_target, f)







        self.normalizer = Normalizer(sample_target) 


    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
            self.config['cuda'] = True
        else:
            device = 'cpu'
            self.config['cuda'] = False
        print("Running on:", device)

        return device

    def group_decay(self, model):
        """Omit weight decay from bias and batchnorm params."""
        decay, no_decay = [], []

        for name, p in model.named_parameters():
            if "bias" in name or "bn" in name or "norm" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        return [
            {"params": decay},
            {"params": no_decay, "weight_decay": 0},
            ]

    def merge_batches(self, *batches):
        merged_data = {}
        cumulative_num_nodes = 0

        for key in batches[0].keys:
            items = []
            for batch in batches:
                item = batch[key]

            # Handle the merging of 'edge_index'.
                if key == 'edge_index':
                    # Offset the edge indices by the cumulative number of nodes.
                    item = item + cumulative_num_nodes
                    items.append(item)
                    # Update the cumulative number of nodes.
                    cumulative_num_nodes += batch.num_nodes
                elif key == 'batch':
                    # Offset the batch tensor by the maximum batch index plus one.
                    max_batch_index = max(items[-1]) if items else -1
                    item = item + max_batch_index + 1
                    items.append(item)
                elif torch.is_tensor(item):
                # Directly concatenate tensors along the first dimension.
                    items.append(item)
                elif isinstance(item, int) or isinstance(item, float):
                # Handle merging of scalar metadata (integers or floats).
                    items.append(item)
                elif isinstance(item, list):
                    # Extend the list if the item is a list.
                    items.extend(item)
                else:
                # Implement other type-specific merging strategies as needed.
                    raise TypeError(f"Unsupported data type: {type(item)}")

            # Concatenate lists or tensors to form the merged data.
            if items and isinstance(items[0], torch.Tensor):
                merged_data[key] = torch.cat(items, dim=0)
            else:
                merged_data[key] = items

        # Create the combined Batch object.
        combined_batch = Batch.from_data_list([Data(**merged_data)])

        return combined_batch
    
    def apply_gaussian_kernel_matrix(self, batch_data, kernel_window):
        """
        Apply the Gaussian kernel to each sample in the batch using matrix operations.
        Replace each element with the maximum weighted value in its window.
        """
        #import ipdb
        #ipdb.set_trace()
        # Convert the kernel window to a tensor
        kernel_tensor = torch.tensor(kernel_window, dtype=torch.float32).cuda()

        padding = (kernel_tensor.size(0) - 1) // 2
        padded_batch = F.pad(batch_data, (padding, padding), mode='reflect')

        
        # Unfold the batch data to create a sliding window view
        unfolded = padded_batch.unfold(1, kernel_tensor.size(0), 1)

        # Apply the Gaussian kernel weights
        weighted_unfolded = unfolded * kernel_tensor

        # Compute the maximum weighted value for each window
        max_values, _ = torch.max(weighted_unfolded, dim=2)

        # Pad the result to match the original shape
        #padding = (kernel_tensor.size(0) - 1) // 2
        #padded_max_values = F.pad(max_values, (padding, padding))

        return max_values

# Apply the Gaussian kernel using matrix operations to the example batch
#matrix_processed_data = apply_gaussian_kernel_matrix(example_data, lds_kernel_window)

#matrix_processed_data




    def train(self, cutoff, neibor,  sigma, beta, thres, loss_rate):
        bin_num = self.normalizer.return_bin()
        model = finetune_ENDE(cutoff, neibor, bin_num)
        #model = self._load_pre_trained_weights(model)
        print("model_loading........")  
        if self.config['cuda']:
            model = model.to(self.device)
        
        layer_list = []
        for name, param in model.named_parameters():
            if 'fc_out' in name:
                print(name, 'new layer')
                layer_list.append(name)
        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))


        #params = self.group_decay(model)
        
        if self.config['optim']['optimizer'] == 'SGD':
            optimizer = optim.SGD(
                [{'params': base_params, 'lr': self.config['optim']['lr']*0.2}, {'params': params}],
                 self.config['optim']['lr'], momentum=self.config['optim']['momentum'], 
                weight_decay=self.config['optim']['weight_decay'])
            
        elif self.config['optim']['optimizer'] == 'Adam':
            optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=self.root_config['epochs'],
            steps_per_epoch=len(self.train_loader),)#20

        else:
            raise NameError('Only SGD or Adam is allowed as optimizer')        
        
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_mae = np.inf
        best_valid_roc_auc = 0
        criterion_spl = SPLLoss(n_samples=len(self.useen_train_loader.dataset), beta=beta, thres=thres)

        for epoch_counter in range(self.root_config['epochs']):
            model.train()
            for bn, (input_1, unseen_input) in enumerate(zip(self.train_loader, self.useen_train_loader)):
                #for bn, (input_1) in enumerate(self.train_loader):
                if self.config['task'] == 'regression':

                    target_normed = self.normalizer.norm(input_1[1].target)
                    target_class, weight = self.normalizer.classify(input_1[1].target)
                    target_class = target_class.cuda()
                else:
                    target_normed = target.view(-1).long()
                    
                target_var = target_normed.cuda()
                weak_input = unseen_input[1]
                strong_input = unseen_input[2]

                # compute output
                output, output_class = model(input_1[1], 'train')
                output_weak, output_weak_class = model(weak_input, 'train')
                #import ipdb
                #ipdb.set_trace()
                weight = self.normalizer.weight_loss(output_weak_class)
                output_strong, output_strong_class = model(strong_input, 'train')


                output=output[:,0]
                output_weak = output_weak[:,0]
                output_strong = output_strong[:,0]
                Lx =F.cross_entropy(output_class, target_class, reduction='mean')
                pseudo_label_class = torch.softmax(output_weak_class.detach(), dim=-1)
                lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=3, sigma=sigma)
                pseudo_label_class = convolve1d(np.array(pseudo_label_class.detach().cpu()), weights=lds_kernel_window, mode='constant')
            
                pseduo_label = output_weak
                max_probs, targets_u = torch.max(torch.from_numpy(pseudo_label_class).cuda(), dim=-1)
                mask  = criterion_spl(output_weak_class, unseen_input[1], weight)
                Lu = torch.mean(torch.abs(output_strong-pseduo_label)*mask)
                optimizer.zero_grad()
                loss = self.criterion(output, target_var)+loss_rate*(Lx+Lu)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    print(epoch_counter, bn, loss.item())
                
                loss.backward()
                optimizer.step()
                print('TRAIN INFO: epoch:{} ({}/{}) iter:{} weight:{:.5f} loss:{:.5f}lxoss:{:.5f}luoss:{:.5f}'.format(epoch_counter, bn + 1, len(self.train_loader), n_iter, max(weight), loss.item(), Lx.item(), Lu.item()))
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['task'] == 'classification': 
                    valid_loss, valid_roc_auc = self._validate(model, self.criterion, self.valid_loader)
                    if valid_roc_auc > best_valid_roc_auc:
                        # save the model weights
                        best_valid_roc_auc = valid_roc_auc
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['task'] == 'regression': 
                    valid_loss ,valid_mae = self._validate(model, self.criterion, self.valid_loader )
                    if valid_mae < best_valid_mae:
                        # save the model weights
                        best_valid_mae = valid_mae
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                    self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                    valid_n_iter += 1
            scheduler.step()
            criterion_spl.increase_threshold()
            print(criterion_spl.threshold)
        self.model = model
        

    def _test_load(self, epoch_num, cutoff, neigbor):
        #fine_tune = FineTune(config, root_config, target_dataset, i_num, current_time)
        loss, metric, metric_rmse, pred, target = self.test(cutoff, neigbor)
        import pandas as pd
        ftf = root_config['fine_tune_from'].split('/')[-1]
        seed = root_config['random_seed']
        fn = '{}_{}_nofine_{}.csv'.format(ftf, task_name,target_dataset)
        print(fn)
        titles= ['num'+str(epoch_num), 'loss', 'mae','rmse']
        df = pd.DataFrame([[i_num, loss, metric.item(), metric_rmse.item()]], columns=titles)
        df.to_csv(os.path.join('experiments', fn), mode='a', index=False)
        df_result = pd.DataFrame(data = {'target': target, 'pred': pred})
        fn_result = os.path.join('experiments', target_dataset)
        _check_file(fn_result)
        df_result.to_csv(os.path.join(fn_result, str(i_num)+fn),  index=False)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(self.root_config['fine_tune_from'], 'checkpoints')
            load_state = torch.load(os.path.join(checkpoints_folder, 'model.pth'),  map_location=self.config['gpu']) 
 
            model_state = model.state_dict()
            for name, param in load_state.items():
                if name not in model_state:
                    print('NOT loaded:', name)
                    continue
                else:
                    print('loaded:', name)
                if isinstance(param, nn.parameter.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                model_state[name].copy_(param)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
        return model

    def _validate(self, model, criterion, valid_loader):
        losses = AverageMeter()
        if self.config['task'] == 'regression':
            mae_errors = AverageMeter()
        else:
            accuracies = AverageMeter()
            precisions = AverageMeter()
            recalls = AverageMeter()
            fscores = AverageMeter()
            auc_scores = AverageMeter()
        model.eval()
        total_loss = []
        val_targets = []
        val_preds = []
        with torch.no_grad():
            #model.eval()

            for bn, (input1) in enumerate(valid_loader):
                target = input1[1].target 
                if self.config['task'] == 'regression':
                    target_normed = self.normalizer.norm(target)
                else:
                    target_normed = target.view(-1).long()
                
                target_var = target_normed.cuda()

                # compute output
                output = model(input1[1])
                output=output[:,0]
                loss = criterion(output, target_var)

                val_pred = self.normalizer.denorm(output.data.cpu())
                val_target = target

                val_preds += val_pred.view(-1).tolist()
                val_targets += val_target.view(-1).tolist()
                
                mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                mae_errors.update(mae_error, target.size(0))
                total_loss.append(loss.item())           
            
            total_loss = sum(total_loss)/len(total_loss)
            mae_result = np.sum(np.abs(np.array(val_preds)-np.array(val_targets)))/len(val_preds)
            print('MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(mae_errors=mae_errors))
            print(mae_result)
            print(total_loss)
        
        model.train()

        if self.config['task'] == 'regression':
            #print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
            return mae_result, mae_result#total_loss, total_loss#losses.avg, mae_errors.avg
        else:
            print('AUC {auc.avg:.3f}'.format(auc=auc_scores))
            return losses.avg, auc_scores.avg

    
    def test(self, cutoff, neibor):
        bin_num = self.normalizer.return_bin()
        test_model = finetune_ENDE(cutoff, neibor, bin_num).cuda()
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        print(model_path)
        
        state_dict = torch.load(model_path, map_location=self.device)
        test_model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        losses = AverageMeter()
        if self.config['task'] == 'regression':
            mae_errors = AverageMeter()
            rmse_errors = AverageMeter()
        else:
            accuracies = AverageMeter()
            precisions = AverageMeter()
            recalls = AverageMeter()
            fscores = AverageMeter()
            auc_scores = AverageMeter()
        
        test_targets = []
        test_preds = []
        test_cif_ids = []

        with torch.no_grad():
            test_model.eval()
            for bn, (input1) in enumerate(self.test_loader):
                target = input1[1].target
                if self.config['task'] == 'regression':
                    target_normed = self.normalizer.norm(target)
                else:
                    target_normed = target.view(-1).long()
                
                target_var = target_normed.cuda()

                # compute output
                output = test_model(input1[1])
                output=output[:,0]
                loss = self.criterion(output, target_var)

                if self.config['task'] == 'regression':
                    mae_error = mae(self.normalizer.denorm(output.data.cpu()), target)
                    rmse_error = rmse(self.normalizer.denorm(output.data.cpu()), target)
                    losses.update(loss.data.cpu().item(), target.size(0))
                    mae_errors.update(mae_error, target.size(0))
                    rmse_errors.update(rmse_error, target.size(0))
                    test_pred = self.normalizer.denorm(output.data.cpu())
                    test_target = target
                    test_preds += test_pred.view(-1).tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += input1[1].cif_id
                else:
                    accuracy, precision, recall, fscore, auc_score = \
                        class_eval(output.data.cpu(), target)
                    losses.update(loss.data.cpu().item(), target.size(0))
                    accuracies.update(accuracy, target.size(0))
                    precisions.update(precision, target.size(0))
                    recalls.update(recall, target.size(0))
                    fscores.update(fscore, target.size(0))
                    auc_scores.update(auc_score, target.size(0))
                   
                    test_pred = torch.exp(output.data.cpu())
                    test_target = target
                    assert test_pred.shape[1] == 2
                    test_preds += test_pred[:, 1].tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += input1[1].cif_id

            mae_result = np.sum(np.abs(np.array(test_preds)-np.array(test_targets)))/len(test_preds)
            rmse_result = np.sqrt( np.sum(np.abs(np.array(test_preds)-np.array(test_targets))**2)/len(test_preds))
            if self.config['task'] == 'regression':
                print('Test: [{0}/{1}], ''Loss {loss.val:.4f} ({loss.avg:.4f}), ''MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    bn, len(self.valid_loader), loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}], '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}), '
                      'Accu {accu.val:.3f} ({accu.avg:.3f}), '
                      'Precision {prec.val:.3f} ({prec.avg:.3f}), '
                      'Recall {recall.val:.3f} ({recall.avg:.3f}), '
                      'F1 {f1.val:.3f} ({f1.avg:.3f}), '
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    bn, len(self.valid_loader), loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))

        with open(os.path.join(self.writer.log_dir, 'test_results.csv'), 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
        
        #self.model.train()

        if self.config['task'] == 'regression':
            print('MAE {mae_errors.avg:.3f}'.format(mae_errors=mae_errors))
            return losses.avg, mae_result, rmse_result, np.array(test_preds), np.array(test_targets)
        else:
            print('AUC {auc.avg:.3f}'.format(auc=auc_scores))
            return losses.avg, auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)
        n = len(tensor)
        range_of_data = torch.max(tensor) - torch.min(tensor)
        sigma = self.std
        bin_width = 3.5 * sigma / np.power(n, 1/3)
        bin_num = int(np.ceil(range_of_data / bin_width))

        self.bin_class = bin_num

        data = tensor.clone().numpy()
        self.bins_class, self.bin_edges_class = pd.cut(data, bins=self.bin_class, retbins=True, labels=range(self.bin_class))
        min_data_value = data.min()
        if self.bin_edges_class[0] >= min_data_value:
                self.bin_edges_class[0] = min_data_value - 1e-10  # 根据数据精度调整这个值
        max_data_value = data.max()
        if self.bin_edges_class[-1] <= max_data_value:
            self.bin_edges_class[-1] = max_data_value + 1e-10  # 根据数据精度调整这个值

        self.bin_num = bin_num
        data = tensor.clone().numpy()
        self.bins, self.bin_edges = pd.cut(data, bins=self.bin_num, retbins=True, labels=range(self.bin_num))
        min_data_value = data.min()
        if self.bin_edges[0] >= min_data_value:
            # 如果最小值正好在第一个边界上或者由于精度问题未被包含，略微扩展这个边界
            self.bin_edges[0] = min_data_value - 1e-10  # 根据数据精度调整这个值

        # 检查并调整最大边界
        max_data_value = data.max()
        if self.bin_edges[-1] <= max_data_value:
            # 如果最大值正好在最后一个边界上或超过，略微扩展这个边界
            self.bin_edges[-1] = max_data_value + 1e-10  # 根据数据精度调整这个值
        num_samples_of_bins = dict(Counter(self.bins))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(max(self.bins)+1)]
        self.eff_label_dist = np.array(emp_label_dist)#convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
        weights = [np.float32(1 / x) for x in self.eff_label_dist]
        max_value_excluding_inf = max(x for x in weights if float(x) != float('inf'))
        weights = [x if float(x)!=float('inf') else max_value_excluding_inf for x in weights]
        self.eff_label_dist = [len(weights) / np.sum(weights) * x for x in weights]#weights* len(weights) / np.sum(weights)
        

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def return_bin(self):
        return self.bin_class

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def classify(self, tensor):
        #import ipdb
        #ipdb.set_trace()
        class_data = tensor.clone().numpy()
        test_bins = pd.cut(class_data, bins=self.bin_edges, right=False, labels=range(self.bin_num))
        #if(test_bins.isna().any()):
        eff_num_per_label = [self.eff_label_dist[bin_idx] for bin_idx in test_bins]
        
        weights = eff_num_per_label#[np.float32(1 / x) for x in eff_num_per_label]
        class_bins =  pd.cut(class_data, bins=self.bin_edges_class, right=False, labels=range(self.bin_class))
        test_bins_tensor = torch.tensor(class_bins, dtype=torch.long)
        weights = torch.tensor(weights, dtype=torch.float32).cuda()
        return test_bins_tensor, weights
    
    def weight_loss(self, tensor):
        test_bins = torch.argmax(F.softmax(tensor, dim=-1), dim=-1)
        eff_num_per_label = [self.eff_label_dist[bin_idx] for bin_idx in test_bins]
        weights = [np.float32(1 / x) for x in eff_num_per_label]
        weights = [len(weights) / np.sum(weights) * x for x in weights]

        

        #print(weights)
        return weights

        

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class Normalizer_maxmin(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.max = torch.max(tensor)
        self.min = torch.min(tensor)

    def norm(self, tensor):
        return (tensor - self.min) /(self.max-self.min)

    def denorm(self, normed_tensor):
        return normed_tensor * (self.max-self.min) + self.min

    def state_dict(self):
        return {'max': self.max,
                'min': self.min}

    def load_state_dict(self, state_dict):
        self.max = state_dict['max']
        self.min = state_dict['min']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))

def rmse(prediction, target):
    return torch.mean(torch.abs(target - prediction)**2)

def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

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
    

if __name__ == "__main__": 
  root_config = yaml.load(open("finetune_config/config_ft.yaml", "r"), Loader=yaml.FullLoader)
    
  target_dataset = root_config["target_dataset"]
  iter_num = root_config['iter_num']
  if target_dataset == 'matbench_dielectric': 
        config = yaml.load(open("finetune_config/config_mb_dielect.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset == 'matbench_jdft2d':
        config = yaml.load(open("finetune_config/config_mb_jdft2d.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset =='matbench_log_kvrh':
        config = yaml.load(open("finetune_config/config_mb_logkv.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset =='matbench_mp_gap':
        config = yaml.load(open("finetune_config/config_mp_gap.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset == 'mp_gap':
        config = yaml.load(open("finetune_config/config_mp_gap.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset == 'mp_bulk':
        config = yaml.load(open("finetune_config/config_mp_bulk.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset == 'mp_shear':
        config = yaml.load(open("finetune_config/config_mp_shear.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset == 'jarvis_formation':
        config = yaml.load(open("finetune_config/config_jarvis_formation.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset == 'jarvis_gap_mbj':
        config = yaml.load(open("finetune_config/config_jarvis_gap_mbj.yaml", "r"), Loader=yaml.FullLoader)
  elif target_dataset == 'jarvis_ehull':
        config = yaml.load(open("finetune_config/config_jarvis_ehull.yaml", "r"), Loader=yaml.FullLoader)
  else:
        config = root_config

  print(config)

  config['task'] =  'regression'
  task_name = target_dataset
  beta =  0.8
  sigma = 0.6
  thres = 0.95
  loss_rate = 0.1
  neibor = 12
  metric_list = []
  rmse_list = []
  cutoff = 8
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True  
  current_time = datetime.now().strftime('%b%d_%H-%M-%S')
  titles= ['num', 'time', 'mae','rmse']
  i_num = 0
  fine_tune = FineTune(config, root_config, target_dataset, i_num, current_time)
  fine_tune.train(cutoff, neibor, sigma, beta, thres, loss_rate)
  loss, metric, metric_rmse, pred, target = fine_tune.test(cutoff, neibor)
  import pandas as pd
  ftf = root_config['fine_tune_from'].split('/')[-1]
  fn = '{}_{}_nofine_{}.csv'.format(ftf, task_name,target_dataset)
  print(fn)
  metric_list.append(metric.item())
  rmse_list.append(metric_rmse.item())
  df = pd.DataFrame([[i_num, loss, metric.item(), metric_rmse.item()]], columns=titles)
  df.to_csv(
            os.path.join('experiments', fn),
            mode='a', index=False
        )
  df_result = pd.DataFrame(data = {'target': target, 'pred': pred})
  fn_result = os.path.join('experiments', target_dataset)
  _check_file(fn_result)
  df_result.to_csv(os.path.join(fn_result, str(i_num)+fn),  index=False)
  import numpy as np
  df = pd.DataFrame([[str(sigma), str(current_time), np.mean(metric_list),np.mean(rmse_list)]], columns=titles)
  df.to_csv(
            os.path.join('experiments', fn),
            mode='a', index=False)
