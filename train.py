import os
import pickle
from subprocess import call

import mxnet as mx
import mxnet.autograd as ag
import mxnet.gluon as gluon
import mxnet.ndarray as nd
import augment
import cfg
from data.cityscapes import CityScapes
from model.lkm import LKM
from model.deeplab import DeepLab
import time

nets = {'LKM': LKM, 'DeepLab': DeepLab}


def train(name, train_loader, ctx, load_checkpoint, learning_rate, num_epochs,
          weight_decay, checkpoint, dropbox):
    records = {'losses': []}
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()
    save_root = os.path.join('save', name)
    if not os.path.exists(save_root):
        call(['mkdir', '-p', save_root])

    loaded = False
    if load_checkpoint:
        save_files = set(os.listdir(save_root))
        if {'weights', 'trainer', 'records'} <= save_files:
            print('Loading checkpoint')
            net = nets[name](pretrained=False)
            net.load_params(os.path.join(save_root, 'weights'), ctx=cfg.ctx)
            trainer = gluon.Trainer(net.collect_params(), 'adam',
                                    {'learning_rate': learning_rate,
                                     'wd': weight_decay})
            trainer.load_states(os.path.join(save_root, 'trainer'))
            with open(os.path.join(save_root, 'records'), 'rb') as f:
                records = pickle.load(f)
            loaded = True
        else:
            print('Checkpoint files don\'t exist.')
            print('Skip loading checkpoint')

    if not loaded:
        net = nets[name](pretrained=True)
        net.collect_params().initialize(mx.initializer.MSRAPrelu(),
                                        ctx=cfg.ctx)
        net.collect_params().reset_ctx(cfg.ctx)
        trainer = gluon.Trainer(net.collect_params(), 'adam', {
                                'learning_rate': learning_rate,
                                'wd': weight_decay})

    net.hybridize()

    print('Start training')
    last_epoch = len(records['losses']) - 1

    for epoch in range(last_epoch + 1, num_epochs):
        iter_count = 0
        t0 = time.time()
        running_loss = 0.0
        for data, label in train_loader:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            batch_size = data.shape[0]
            with ag.record(train_mode=True):
                output = net(data)
                output = nd.transpose(output, axes=(0, 2, 3, 1))
                loss = criterion(output, label)
            loss.backward()
            trainer.step(batch_size)

            _loss = nd.sum(loss).asscalar() / batch_size
            print('\rEpoch {} Iter {} Loss {:.4f}'.format(
                epoch, iter_count, _loss), end='')
            iter_count += 1

            running_loss += _loss

        t1 = time.time()
        print('\rEpoch {} : Loss {:.4f}  Time {:.2f}min'.format(
            epoch, running_loss, (t1 - t0) / 60))
        records['losses'].append(running_loss)

        if (epoch + 1) % checkpoint == 0:
            print('\rSaving checkpoint', end='')
            net.save_params(os.path.join(save_root, 'weights'))
            trainer.save_states(os.path.join(save_root, 'trainer'))
            with open(os.path.join(save_root, 'records'), 'wb') as f:
                pickle.dump(records, f)
            if dropbox:
                call(['cp', '-r', save_root,
                      os.path.join(cfg.home, 'Dropbox')])
            print('\rFinish saving checkpoint', end='')
    print('\nFinish training')


def main():
    train_dataset = CityScapes(
        cfg.cityscapes_root, 'train', augment.cityscapes_train)
    train_loader = gluon.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True)
    train('LKM', train_loader, cfg.ctx, True, 0.00015, 100, 0.0001, 1, True)


if __name__ == '__main__':
    main()
