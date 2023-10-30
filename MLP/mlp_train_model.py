import torch
import torch.nn as nn

import sys
sys.path.append('../LGG/')

from utils import *
from train_model import *


CUDA = torch.cuda.is_available()

def mlp_set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.mlp_save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def train_simpleMLP(model, args, data_train, label_train, data_val, label_val, subject):
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_mlp'
    mlp_set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)

    val_loader = get_dataloader(data_val, label_val, args.batch_size)

    # model = get_model(args)
    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()


    def save_model(name):
        previous_model = osp.join(args.mlp_save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.mlp_save_path, '{}.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['F1'] = 0.0

    timer = Timer()
    patient = args.patient
    counter = 0

    for epoch in range(1, args.max_epoch + 1):

        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))

        loss_val, pred_val, act_val = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn
        )
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))

        if acc_val >= trlog['max_acc']:
            trlog['max_acc'] = acc_val
            trlog['F1'] = f1_val
            save_model('candidate')
            counter = 0
        else:
            counter += 1
            if counter >= patient:
                print('early stopping')
                break

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        print('ETA:{}/{} SUB:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                 subject))
    # save the training log file
    save_name = 'trlog' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    save_path = osp.join(args.mlp_save_path, experiment_setting, 'log_train')
    ensure_path(save_path)
    torch.save(trlog, osp.join(save_path, save_name))

    return trlog['max_acc'], trlog['F1']


def test_simpleMLP(model, args, data, label, reproduce, subject):
    mlp_set_up(args)
    seed_all(args.random_seed)
    test_loader = get_dataloader(data, label, args.batch_size)

    # model = get_model(args)
    if CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    if reproduce:
        model_name_reproduce = 'sub' + str(subject) + '_mlp' + '.pth'
        data_type = 'model_{}_{}'.format(args.data_format, args.label_type)
        experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
        load_path_final = osp.join(args.mlp_save_path, experiment_setting, data_type, model_name_reproduce)
        model.load_state_dict(torch.load(load_path_final))
    else:
        model.load_state_dict(torch.load(args.mlp_load_path_final))
    loss, pred, act = predict(
        data_loader=test_loader, net=model, loss_fn=loss_fn
    )
    acc, f1, cm = get_metrics(y_pred=pred, y_true=act)
    print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))
    return acc, pred, act


def train_complexMLP(model, args, data_train, label_train, data_val, label_val, subject, alpha=1):
    '''
    alpha: trade of between two loss terms --> loss = loss_fn + alpha * loss_2
    '''
    seed_all(args.random_seed)
    save_name = '_sub' + str(subject) + '_mlp'
    mlp_set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)

    val_loader = get_dataloader(data_val, label_val, args.batch_size)

    # model = get_model(args)
    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    ### binary classification prediction loss
    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()
    ### facial embedding prediction loss
    loss_2 = nn.MSELoss()


    def save_model(name):
        previous_model = osp.join(args.mlp_save_path, '{}_cmlp.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.mlp_save_path, '{}_cmlp.pth'.format(name)))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['F1'] = 0.0

    timer = Timer()
    patient = args.patient
    counter = 0

    for epoch in range(1, args.max_epoch + 1):

        loss_bi_train, loss_mse_train, loss_total_train, pred_train, act_train = train_one_epoch_cmlp(
            data_loader=train_loader, net=model, loss_fn=loss_fn, loss_2=loss_2, alpha=alpha, optimizer=optimizer)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('epoch {}, loss_bi={:.4f} loss_mse={:.4f} loss_total={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_bi_train, loss_mse_train, loss_total_train, acc_train, f1_train))

        loss_bi_val, loss_mse_val, loss_total_val, pred_val, act_val = predict_cmlp(
            data_loader=val_loader, net=model, loss_fn=loss_fn, loss_2=loss_2, alpha=alpha
        )
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, val, loss_bi={:.4f} loss_mse={:.4f} loss_total={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_bi_val, loss_mse_val, loss_total_val, acc_val, f1_val))

        if acc_val >= trlog['max_acc']:
            trlog['max_acc'] = acc_val
            trlog['F1'] = f1_val
            save_model('candidate')
            counter = 0
        else:
            counter += 1
            if counter >= patient:
                print('early stopping')
                break

        trlog['train_loss'].append(loss_total_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_total_val)
        trlog['val_acc'].append(acc_val)

        print('ETA:{}/{} SUB:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                 subject))
    # save the training log file
    save_name = 'trlog' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    save_path = osp.join(args.mlp_save_path, experiment_setting, 'log_train')
    ensure_path(save_path)
    torch.save(trlog, osp.join(save_path, save_name))

    return trlog['max_acc'], trlog['F1']


def test_complexMLP(model, args, data, label, reproduce, subject, alpha=1):
    mlp_set_up(args)
    seed_all(args.random_seed)
    test_loader = get_dataloader(data, label, args.batch_size)

    # model = get_model(args)
    if CUDA:
        model = model.cuda()
    ### binary classification prediction loss
    loss_fn = nn.CrossEntropyLoss()
    ### facial embedding prediction loss
    loss_2 = nn.MSELoss()

    if reproduce:
        model_name_reproduce = 'sub' + str(subject) + '_cmlp' + '.pth'
        data_type = 'model_{}_{}'.format(args.data_format, args.label_type)
        experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
        load_path_final = osp.join(args.mlp_save_path, experiment_setting, data_type, model_name_reproduce)
        model.load_state_dict(torch.load(load_path_final))
    else:
        model.load_state_dict(torch.load(args.mlp_load_path_final))
    loss_bi, loss_mse, loss_total, pred, act = predict_cmlp(
        data_loader=test_loader, net=model, loss_fn=loss_fn, loss_2=loss_2, alpha=alpha
    )
    acc, f1, cm = get_metrics(y_pred=pred, y_true=act)
    print('>>> Test:  loss_bi={:.4f} loss_mse={:.4f} loss_total={:.4f} acc={:.4f} f1={:.4f}'\
              .format(loss_bi, loss_mse, loss_total, acc, f1))
    return acc, pred, act


def train_one_epoch_cmlp(data_loader, net, loss_fn, loss_2, alpha, optimizer):
    net.train()
    tl, tl_mse, tl_total = Averager(), Averager(), Averager()

    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        out, face_embed_pred, face_embed_low = net(x_batch)
        loss1 = loss_fn(out, y_batch)
        loss2 = loss_2(face_embed_pred, face_embed_low)
        loss_total = loss1 + alpha * loss2
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        tl.add(loss1.item())
        tl_mse.add(loss_total.item())
        tl_total.add(loss2.item())
    return tl.item(), tl_mse.item(), tl_total.item(), pred_train, act_train


def predict_cmlp(data_loader, net, loss_fn, loss_2, alpha, ):
    net.eval()
    pred_val = []
    act_val = []
    vl, vl_mse, vl_total = Averager(), Averager(), Averager()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            out, face_embed_pred, face_embed_low = net(x_batch)
            loss1 = loss_fn(out, y_batch)
            loss2 = loss_2(face_embed_pred, face_embed_low)
            loss_total = loss1 + alpha * loss2
            _, pred = torch.max(out, 1)
            vl.add(loss1.item())
            vl_mse.add(loss2.item())
            vl_total.add(loss_total.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), vl_mse.item(), vl_total.item(), pred_val, act_val