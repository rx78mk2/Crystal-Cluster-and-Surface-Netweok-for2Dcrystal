import json
import time
import sys

import pandas as pd
import torch
import shutil
from torch.autograd import Variable

def trainmodel(model, criterion, optimizer, scheduler, normalizer,
                 train_loader, val_loader, args, best_mae_error, fold=0):

    mae_list = []
    best_epoch = 0
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, normalizer, args)

        mae_error = validate(val_loader, model, criterion, normalizer, args)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        mae_list.append(float(mae_error.numpy()))
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        if(is_best == True):
            best_epoch = epoch

        checkpoint_name = args.program_name + '_checkpoint_'+str(fold)+'_.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best, checkpoint_name)

    outfile = 'val_result' + str(fold) + '.xlsx'
    mae_error = validate(val_loader, model, criterion, normalizer, args, outfile=outfile, test=True)
    print("Best MAE  is : ", best_mae_error, "of epoch:", best_epoch)

    return mae_error, mae_list, best_mae_error



'''###------------------------------------------------------------------------------------------------------------###'''


def train(train_loader, model, criterion, optimizer, epoch, normalizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    mae_errors = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target_normed = normalizer.norm(target)
        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True), input[3].cuda(non_blocking=True),
                         input[4].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[5]])
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2], input[3],
                         input[4],
                         [crys_idx for crys_idx in input[5]])
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, mae_errors=mae_errors)
            )


'''###------------------------------------------------------------------------------------------------------------###'''


def validate(val_loader, model, criterion, normalizer, args, outfile='test_results.csv', test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []
        test_preds_nor = []
        test_targets_nor = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):

        target_normed = normalizer.norm(target)
        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True), input[3].cuda(non_blocking=True),
                         input[4].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[5]])
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2], input[3],
                         input[4],
                         [crys_idx for crys_idx in input[5]])
            target_var = Variable(target_normed)

        # compute output
        target_normed = normalizer.norm(target)
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_pred_nor = output.data.cpu()
            test_target_nor = target_normed
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids
            test_preds_nor += test_pred_nor.view(-1).tolist()
            test_targets_nor += test_target_nor.view(-1).tolist()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                mae_errors=mae_errors))


    if test:
        star_label = '**'

        output_result = {}
        output_result['ID'] = test_cif_ids
        output_result['target'] = test_targets
        output_result['predict'] = test_preds
        output_result['targer_nor'] = test_targets_nor
        output_result['predict_nor'] = test_preds_nor

        outputxlsx = pd.DataFrame(output_result)
        outputxlsx.to_excel(outfile)

    else:
        star_label = '*'

    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label, mae_errors=mae_errors))
    return mae_errors.avg


'''###------------------------------------------------------------------------------------------------------------###'''


def validate_pre(val_loader, model, criterion, normalizer, args, fold=0, outfile='test_results.csv', test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []
        test_preds_nor = []
        test_targets_nor = []

    fea_2to3s = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):

        target_normed = normalizer.norm(target)
        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True), input[3].cuda(non_blocking=True),
                         input[4].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[5]])
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2], input[3],
                         input[4],
                         [crys_idx for crys_idx in input[5]])
            target_var = Variable(target_normed)

        # compute output
        target_normed = normalizer.norm(target)
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_pred_nor = output.data.cpu()
            test_target_nor = target_normed
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids
            test_preds_nor += test_pred_nor.view(-1).tolist()
            test_targets_nor += test_target_nor.view(-1).tolist()



        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                mae_errors=mae_errors))


    if test:
        star_label = '**'

        output_result = {}
        output_result['ID'] = test_cif_ids
        output_result['target'] = test_targets
        output_result['predict'] = test_preds
        output_result['targer_nor'] = test_targets_nor
        output_result['predict_nor'] = test_preds_nor

        outputxlsx = pd.DataFrame(output_result)
        outputxlsx.to_excel(outfile)

    else:
        star_label = '*'

    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label, mae_errors=mae_errors))
    return mae_errors.avg

'''###------------------------------------------------------------------------------------------------------------###'''


def MiddleLayerOutput(model, input_var, layer_name = 'Fc1'):
    # 输出中间层的输出测试部分
    features_in_hook = []
    features_out_hook = []

    def hook(module, fea_in, fea_out):
        features_in_hook.append(fea_in)
        features_out_hook.append(fea_out)
        return None


    for (name, module) in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook=hook)

    # compute output
    output = model(*input_var)


    out = features_out_hook

    out = out[0]

    return out


'''###------------------------------------------------------------------------------------------------------------###'''

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


'''###------------------------------------------------------------------------------------------------------------###'''


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


'''###------------------------------------------------------------------------------------------------------------###'''


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    best_filename = 'best_' + filename
    if is_best:
        shutil.copyfile(filename, best_filename)


'''###------------------------------------------------------------------------------------------------------------###'''