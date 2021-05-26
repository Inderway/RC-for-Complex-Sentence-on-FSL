from

device = torch.device('cuda:0') if torch.cuda.is_available() and opt.cuda else torch.device('cpu')