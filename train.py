from include import *
from utils import *
from model import SEResNext50
from efficientnet_pytorch import EfficientNet
from datasets import IntraDataset, aug_image, read_trainset, read_testset

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

train_df = read_trainset(DATA_DIR + "stage_2_train.csv")
test_df = read_testset(DATA_DIR + "stage_2_test.csv")


def run_train(checkpoint=None):  

    train_dataset = IntraDataset(
        df = train_df.loc[train_idx],
        load_image_function=load_ssb_dicom,
        augment = None
    )

    train_loader  = DataLoader(
        train_dataset,
        batch_size  = 16,
        shuffle=True,
        drop_last   = True,
        num_workers = 4,
        pin_memory  = True
    )
    
    valid_dataset = IntraDataset(
        df = train_df.loc[val_idx],
        load_image_function=load_ssb_dicom,
        augment = None,
    )
    valid_loader  = DataLoader(
        valid_dataset,
        batch_size  = 16,
        shuffle=True,
        drop_last   = True,
        num_workers = 4,
        pin_memory  = True
    )
    if checkpoint is None:
        start_epoch = 0
        model = SEResNext50()
        # model = EfficientNet.from_pretrained('efficientnet-b0')
        # model._fc = torch.nn.Linear(1280,n_classes)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
    epochs = 3
    
    model.to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train_one_epoch(train_loader=train_loader,
                        valid_loader = valid_loader,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        epoch=epoch)
        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)
    del model
        
        
        
def evaluate(valid_loader,model,criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        
        for t,(inputs, truth_label) in enumerate(tqdm(valid_loader)):
            inputs = inputs.to(device,dtype=torch.float)
            truth_label = truth_label.to(device,dtype=torch.float)
            logit_label = model(inputs)
            loss = criterion(logit_label, truth_label)
            val_loss += loss.item()
            
        val_loss = val_loss/len(valid_loader)
        
        print('Valid Loss: {:.4f} '.format(val_loss))
        
    
def train_one_epoch(train_loader,valid_loader,model, criterion, optimizer, epoch):
        
    print('Epoch {}'.format(epoch))
    print('-' * 10)
    
    model.train()
    tr_loss   = 0
    
    start_time = time.time()

    for t,(inputs, truth_label) in enumerate(tqdm(train_loader)):
        inputs = inputs.to(device,dtype=torch.float)
        truth_label = truth_label.to(device,dtype=torch.float)
        
        optimizer.zero_grad()

        logit_label = model(inputs)
        loss = criterion(logit_label, truth_label)
        
        tr_loss += loss.item()
        loss.backward()
        optimizer.step()
        print('Training batch Loss: {:.4f}'.format(loss.item()))
        del inputs,truth_label,logit_label,loss

    epoch_loss = tr_loss/len(train_loader)

    print('Training epoch Loss: {:.4f} with time: {:.3f}'.format(epoch_loss,time.time()-start_time))

    evaluate(valid_loader,model,criterion)

if __name__ == '__main__':
    run_train()