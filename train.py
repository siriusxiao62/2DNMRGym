import torch
import torch.nn as nn
import time

def train_model(model, dataloaders, optimizer, scheduler, checkpoint_path, num_epochs=1):
    best_loss = 1e10

    model = model.cuda()

    stalled = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0

            # print(len(dataloaders[phase]))
            for graph in dataloaders[phase]:
                # graph, cnmr, hnmr, filename = batch
                # print(filename)
                graph = graph.cuda()
                # cnmr = cnmr.cuda()
                # hnmr = hnmr.cuda()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    [c_shifts, h_shifts], c_idx = model(graph)
                    
                    loss = nn.MSELoss()(c_shifts, graph.cnmr) + nn.MSELoss()(h_shifts, graph.hnmr)
                    loss *= 100
                    epoch_loss += loss
                    # print(loss)
                    if torch.isnan(loss):
                        print(graph.filename)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
            epoch_loss = epoch_loss / (len(dataloaders[phase]))
            print(phase + 'loss', epoch_loss)
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

            # save the model weights
            if phase == 'val':
                if epoch_loss < best_loss:
                    stalled = 0
                    print(f"saving best model to {checkpoint_path}")
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), checkpoint_path)
                else:
                    stalled += 1
        if stalled >10:
            print('stopped trainig early at epoch: ', epoch)
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # save the last trained model
    # torch.save(model.state_dict(), 'final_%s'%checkpoint_path)

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model