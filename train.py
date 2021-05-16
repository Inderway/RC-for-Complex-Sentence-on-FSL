from data_loader import get_data_loader
from FSMRE import FSMRE
from FSMRE_loss import get_loss as loss_fn
from parser_util import get_parser

from transformers import BertModel
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import os
from torch_geometric.nn import GCNConv



def init_seed(opt):
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

def init_dataloader(opt, mode):
    # todo: alter parameters
    data_loader=get_data_loader(opt.dataset_root, opt.classes_per_it_tr, opt.batch_num, 1, 1,mode)
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
    # FixMe encoder
    encoder=BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
    # FixMe aggragator
    aggregator = nn.LSTM(768, opt.hidden_dim, bidirectional=True)
    # FixMe propagator
    propagator=GCNConv(opt.hidden_dim, opt.hidden_dim)
    model = FSMRE(encoder=encoder, aggregator=aggregator, propagator=propagator)#.to(device)
    return model

def train(opt, dataloader, model, optim, lr_scheduler):
    device='cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    train_loss=[]
    train_single_acc=[]
    train_multi_acc=[]
    best_multi_acc=0
    best_state=None
    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter=iter(dataloader)

        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            support_set, query_set, labels=batch
            support_set=support_set[0]
            query_set=query_set[0]
            labels=labels[0]
            # support_set, query_set=support_set.to(device), query_set.to(device)
            label_num=len(labels)
            # sentence_num*entity_num*entity_num*label_num
            model_output=model(support_set, query_set,labels)

            loss, single_acc, multi_acc=loss_fn(model_output, query_set[4], label_num)

            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_single_acc.append(single_acc)
            train_multi_acc.append(multi_acc)
        avg_loss=np.mean(train_loss[-opt.batch_num:])
        avg_single_acc=np.mean(train_single_acc[-opt.batch_num:])
        avg_multi_acc=np.mean(train_multi_acc[-opt.batch_num:])
        print('Avg Train Loss: {}, Avg Train Single Acc: {}, Avg Train Multi Acc: {}'.format(avg_loss, avg_single_acc, avg_multi_acc))
        lr_scheduler.step()
        if avg_multi_acc>best_multi_acc:
            torch.save(model.state_dict(), best_model_path)
            best_multi_acc=avg_multi_acc
            best_state=model.state_dict()

    torch.save(model.state_dict(), last_model_path)
    for name in ['train_loss', 'train_single_acc', 'train_multi_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])
    return best_state, best_multi_acc, train_loss, train_single_acc, train_multi_acc

def test(opt, test_dataloader, model):
    single_acc = []
    multi_acc = []
    for epoch in range(10):
        test_iter=iter(test_dataloader)
        for batch in tqdm(test_iter):
            support_set, query_set, labels=batch
            support_set=support_set[0]
            query_set=query_set[0]
            labels=labels[0]
            # support_set, query_set=support_set.to(device), query_set.to(device)
            label_num=len(labels)
            # sentence_num*entity_num*entity_num*label_num
            model_output=model(support_set, query_set,labels)

            _, single_acc, multi_acc=loss_fn(model_output, query_set[4], label_num)



            single_acc.append(single_acc)
            multi_acc.append(multi_acc)
    avg_single_acc=np.mean(single_acc)
    avg_multi_acc=np.mean(multi_acc)
    print('Avg Single Acc: {}, Avg Multi Acc: {}'.format(avg_single_acc, avg_multi_acc))
    return avg_multi_acc

def eval(opt, test_dataloader):
    options=get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    model=init_model(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

def main():
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)
    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    tr_dataloader=init_dataloader(options, 'train')
    test_dataloader=init_dataloader(options, 'test')
    model=init_model(options)
    optim=init_optim(options,model)
    lr_scheduler=init_lr_scheduler(options, optim)
    res=train(opt=options,
              dataloader=tr_dataloader,
              model=model,
              optim=optim,
              lr_scheduler=lr_scheduler)
    best_state, best_multi_acc, train_loss, train_single_acc, train_multi_acc= res
    print('Testing with last model================')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)
    model.load_state_dict(best_state)
    print('Testing with best model================')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

if __name__ == '__main__':
    print("Begin")
    main()