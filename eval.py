import torch
import time
import torch.nn.functional as F



def eval_model(model, dataloader):
    model = model.cuda()

    since = time.time()
    model.eval()

    total_loss_c = 0
    total_loss_h = 0
    print(len(dataloader))
    with torch.no_grad():
        for graph in dataloader:
            graph = graph.cuda()

            with torch.cuda.amp.autocast():
                try:
                    [c_shifts, h_shifts], c_idx = model(graph)
                except:
                    continue
 
                c_loss = F.l1_loss(c_shifts, graph.cnmr) * 200
                h_loss = F.l1_loss(h_shifts, graph.hnmr) * 10
                total_loss_c += c_loss
                total_loss_h += h_loss
            # print(loss)
                
        total_loss_c = total_loss_c / (len(dataloader))
        total_loss_h = total_loss_h / (len(dataloader))

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Val C loss: {:4f}, H loss: {:4f}'.format(total_loss_c, total_loss_h))

    return total_loss_c, total_loss_h




    