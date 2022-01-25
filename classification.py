from utils.datasets import ClsDataset
from utils.preprocessing import *
from utils.utils import *
from utils.model import ClsModel
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from time import sleep
from tqdm import tqdm

device = 'cuda'
image_dir = './dataset/classification/train'
mask_dir = './dataset/classification/output_mask'
df = load_csv('./dataset/train.csv')

train, rest = train_test_split(df, test_size=0.4, random_state=42)
valid, test = train_test_split(rest, test_size=0.5, random_state=42)

train = upsampling(train)

train_dataset = ClsDataset(train, image_dir, mask_dir, augmentation=transforms_train, preprocessing=get_preprocessing())
valid_dataset = ClsDataset(valid, image_dir, mask_dir, augmentation=transforms_val, preprocessing=get_preprocessing())
test_dataset = ClsDataset(test, image_dir, mask_dir, augmentation=transforms_val, preprocessing=get_preprocessing())

model = ClsModel('tf_efficientnet_b1_ns', 3)
model = model.to(device)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

ce = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

max_acc = 0
for epoch in range(1, 40):
    train_loss = 0
    train_acc = 0
    valid_loss = 0
    valid_acc = 0
    i = 0

    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        for data, target in tepoch:
            i += 1
            tepoch.set_description(f"Epoch {epoch}")

            data, target = data.to(device), target.to(device).long().squeeze()
            optimizer.zero_grad()
            output = model(data)
            target = target.argmax(dim=1, keepdim=True).squeeze()
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            loss = ce(output, target)
            train_loss += loss.item()
            correct = (predictions == target).sum().item()
            train_acc += correct / 4

            loss.backward()
            optimizer.step()
            
            tepoch.set_postfix(loss=train_loss/i, accuracy=train_acc/i)
            sleep(0.1)
    
    model.eval()
    i = 0
    with tqdm(valid_loader, unit="batch") as tepoch:
        for data, target in tepoch:
            with torch.no_grad():
                i += 1
                tepoch.set_description(f"Epoch {epoch}")

                data, target = data.to(device), target.to(device).long().squeeze()
                output = model(data)
                target = target.argmax(dim=0, keepdim=True)
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                loss = ce(output, target)
                valid_loss += loss.item()
                correct = (predictions == target).sum().item()
                valid_acc += correct / 1
                
                tepoch.set_postfix(loss=valid_loss/i, accuracy=valid_acc/i)
                sleep(0.1)
    
    if valid_acc > max_acc:
        max_acc = valid_acc
        torch.save(model, './model/' + 'ClsModel.pth')
        print('Model Saved!')