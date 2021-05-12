from data_loader import get_data_loader
from FSMRE import FSMRE
from FSMRE_loss import get_loss as loss_fn
from parser_util import get_parser

import numpy as np
from tqdm import tqdm
import torch
import os



def init_seed(opt):
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

def init_dataloader(opt, mode):
    # todo: alter parameters
    data_loader=get_data_loader(opt.dataset_root, opt.classes_per_it_tr, opt.batch_num, 25, 10)
    return data_loader

def init_optim(opt,model):
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)

def init_lr_scheduler(opt, optim):
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)

def init_model(opt):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    # todo: alter the parameters, add encoder, aggregator and propagator
    model = FSMRE().to(device)
    return model

def train(opt, dataloader, model, optim, lr_scheduler):
    device='cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    train_loss=[]
    train_acc=[]
    val_loss=[]
    val_acc=[]
    best_acc=0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter=iter(dataloader)

        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            support_set, query_set, labels=batch
            support_set, query_set=support_set.to(device), query_set.to(device)
            label_num=len(labels)
            # sentence_num*entity_num*entity_num*label_num
            model_output=model(support_set, query_set)
            loss=loss_fn(model_output, query_set[4], label_num)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
        avg_loss=np.mean(train_loss[-opt.batch_num:])
        print('Avg Train Loss: {}'.format(avg_loss))
        lr_scheduler.step()



    torch.save(model.state_dict(), last_model_path)

    return model.state_dict(), train_loss

def eval(opt):
    options=get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    model=init_model(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

def main():
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)
    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    tr_dataloader=init_dataloader(options, 'train')
    model=init_model(options)
    optim=init_optim(options,model)
    lr_scheduler=init_lr_scheduler(options, optim)
    res=train(opt=options,
              dataloader=tr_dataloader,
              model=model,
              optim=optim,
              lr_scheduler=lr_scheduler)
    state, train_loss=res
    print('result================')
    print(train_loss)

if __name__ == '__main__':
    main()