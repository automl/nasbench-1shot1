import os
import json
import logging

from nasbench1shot1.optimizers.oneshot.darts.searcher import DARTSWrapper
from nasbench1shot1.optimizers.oneshot.base import Architect


class DARTS(object):
    def __init__(self, args, search_space):
        self.args = args
        self.model = DARTSWrapper(args, search_space)
        self.architect = Architect(self.model.model, args)

    def run(self):
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        # increase the cutout probability linearly throughout search
        train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
        logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                     train_transform.transforms[-1].cutout_prob)

        # Save the one shot model architecture weights for later analysis
        arch_filename = os.path.join(args.save, 'one_shot_architecture_{}.obj'.format(epoch))
        with open(arch_filename, 'wb') as filehandler:
            numpy_tensor_list = []
            for tensor in model.arch_parameters():
                numpy_tensor_list.append(tensor.detach().cpu().numpy())
            pickle.dump(numpy_tensor_list, filehandler)

        # Save the entire one-shot-model
        filepath = os.path.join(args.save, 'one_shot_model_{}.obj'.format(epoch))
        torch.save(model.state_dict(), filepath)

        logging.info('architecture', numpy_tensor_list)

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


