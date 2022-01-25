import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from utils.datasets import SegDataset
from utils.model import SegModel
from utils.preprocessing import *

DEVICE = 'cuda'

x_train_dir = 'D:/Kaggle/dataset/train'
y_train_dir = 'D:/Kaggle/dataset/train_annot'

x_valid_dir = 'D:/Kaggle/dataset/valid'
y_valid_dir = 'D:/Kaggle/dataset/valid_annot'

x_test_dir = 'D:/Kaggle/dataset/test'
y_test_dir = 'D:/Kaggle/dataset/test_annot'

train_dataset = SegDataset(x_train_dir, y_train_dir, augmentation=transforms_train, preprocessing=get_preprocessing())
valid_dataset = SegDataset(x_valid_dir, y_valid_dir, augmentation=transforms_val, preprocessing=get_preprocessing())
test_dataset = SegDataset(x_test_dir, y_test_dir, augmentation=transforms_val, preprocessing=get_preprocessing())

model = SegModel('timm-efficientnet-b1')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

loss = smp.utils.losses.DiceLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]
    
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

max_score = 0

best_model = None

for i in range(0, 20):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, 'D:/Kaggle/model/' + 'SegModel2.pth')
        best_model = model
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

test_dataloader = DataLoader(test_dataset)

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)