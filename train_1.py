from data_loader import get_data_loader
from FSMRE import FSMRE
from FSMRE_loss import get_loss as loss_fn
from parser_util import get_parser

from transformers import BertModel
import numpy as np
import torch
import torch.nn as nn
import os
from torch_geometric.nn import GCNConv
import traceback


def init_seed(opt):
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

def init_dataloader(opt, mode):
    # todo: alter parameters
    path='data'+os.sep+opt.name
    data_loader=get_data_loader(path, opt.classes_per_it_tr, opt.batch_num, opt.num_support_tr, opt.num_query_tr,mode)
    return data_loader

def init_optim(opt,model):
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)

def init_lr_scheduler(opt, optim):
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)

def init_model(opt):
    device = torch.device('cuda:1') if torch.cuda.is_available() and opt.cuda else torch.device('cpu')
    # FixMe encoder
    encoder=BertModel.from_pretrained('bert-base-cased', output_hidden_states=True).to(device)
    # FixMe aggragator
    aggregator = nn.LSTM(768, opt.hidden_dim, bidirectional=True).to(device)
    # FixMe propagator
    propagator=GCNConv(opt.hidden_dim, opt.hidden_dim).to(device)
    model = FSMRE(encoder=encoder, aggregator=aggregator, propagator=propagator, device=device).to(device)
    return model

def train(opt, dataloader, model, optim, lr_scheduler):
    device=torch.device('cuda:1') if torch.cuda.is_available() and opt.cuda else torch.device('cpu')
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
        for b_id, batch in enumerate(tr_iter):
            print('--- batch: {} ---'.format(b_id))
            optim.zero_grad()
            support_set, query_set, label_num=batch
            support_set=support_set[0]
            query_set=query_set[0]
            label_num=label_num[0]

            input_support_set=[]
            input_support_set.append(support_set[0].to(device))
            input_support_set.append(support_set[1].to(device))
            for i in range(3):
                input_support_set.append([])
            for i in range(2,5):
                for ele in support_set[i]:
                    input_support_set[i].append(ele.to(device))

            input_query_set = []
            input_query_set.append(query_set[0].to(device))
            input_query_set.append(query_set[1].to(device))
            for i in range(3):
                input_query_set.append([])
            for i in range(2, 5):
                for ele in query_set[i]:
                    input_query_set[i].append(ele.to(device))
            
            model_output=None
            try:
                model_output=model(input_support_set, input_query_set,label_num)
            except Exception:
                print(input_support_set)
                print(">>>>>>>>>>>>>>>>>>")
                print(input_query_set)
                traceback.print_exc()
                assert 0
            loss, single_acc, multi_acc=loss_fn(model_output, input_query_set[4], label_num)

            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_single_acc.append(single_acc)
            train_multi_acc.append(multi_acc)
        avg_loss=np.mean(train_loss)
        avg_single_acc=np.mean(train_single_acc)
        avg_multi_acc=np.mean(train_multi_acc)
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
    device = torch.device('cuda:1') if torch.cuda.is_available() and opt.cuda else torch.device('cpu')
    single_acc_l = []
    multi_acc_l = []
    for epoch in range(5):
        print('=== Epoch: {} ==='.format(epoch))
        test_iter=iter(test_dataloader)
        for b_id, batch in enumerate(test_iter):
            print('--- batch: {} ---'.format(b_id))
            support_set, query_set, label_num=batch
            support_set=support_set[0]
            query_set=query_set[0]
            # support_set, query_set=support_set.to(device), query_set.to(device)
            label_num=label_num[0]
            # sentence_num*entity_num*entity_num*label_num
            input_support_set = []
            input_support_set.append(support_set[0].to(device))
            input_support_set.append(support_set[1].to(device))
            for i in range(3):
                input_support_set.append([])
            for i in range(2, 5):
                for ele in support_set[i]:
                    input_support_set[i].append(ele.to(device))

            input_query_set = []
            input_query_set.append(query_set[0].to(device))
            input_query_set.append(query_set[1].to(device))
            for i in range(3):
                input_query_set.append([])
            for i in range(2, 5):
                for ele in query_set[i]:
                    input_query_set[i].append(ele.to(device))

            model_output=None
            try:
                model_output=model(input_support_set, input_query_set,label_num)
            except Exception:
                print(input_support_set)
                print(">>>>>>>>>>>>>>>>>>")
                print(input_query_set)
                traceback.print_exc()
                assert 0
            model_output=model(input_support_set, input_query_set,label_num)

            _, single_acc, multi_acc=loss_fn(model_output, input_query_set[4], label_num)



            single_acc_l.append(single_acc)
            multi_acc_l.append(multi_acc)
    avg_single_acc=np.mean(single_acc_l)
    avg_multi_acc=np.mean(multi_acc_l)
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