from include import *
from datasets import IntraDataset, read_testset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = 'checkpoint.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

def run_inference(model):
    n_classes = 6
    batch_size = 16
    test_pred = np.zeros((len(test_df) * 6, 1))
    
    test_dataset = IntraDataset(
        df = test_df,
        load_image_function=load_ssb_dicom,
        augment = None,
        mode = "test",
        img_dir = TEST_IMAGES_DIR
    )
    test_loader  = DataLoader(
        test_dataset,
        batch_size  = 16,
        shuffle=True,
        drop_last   = True,
        num_workers = 4,
        pin_memory  = True
    )
    model.eval()
    for i, inputs in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device, dtype=torch.float)

        with torch.no_grad():
            pred = model(inputs)

            test_pred[(i * batch_size * n_classes):((i + 1) * batch_size * n_classes)] = torch.sigmoid(
                pred).detach().cpu().reshape((len(inputs) * n_classes, 1))
    
    submission =  pd.read_csv(os.path.join(DATA_DIR, 'stage_2_sample_submission.csv'))
    submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)
    submission.columns = ['ID', 'Label']

    submission.to_csv('submission.csv', index=False) 
    
         
if __name__ == '__main__':
    run_inference()

