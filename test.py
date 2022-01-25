from utils.datasets import ClsDataset
from utils.preprocessing import *
from utils.utils import *
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

test_dataset = ClsDataset(test, image_dir, mask_dir, augmentation=transforms_val, preprocessing=get_preprocessing())

model = torch.load('./model/ClsModel.pth')
model = model.to(device)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

ce = nn.CrossEntropyLoss()

i = 0
test_loss = 0
test_acc = 0

model.eval()

with tqdm(test_loader, unit="batch") as tepoch:
    for data, target in tepoch:
        with torch.no_grad():
            i += 1
            tepoch.set_description(f"Test:")

            data, target = data.to(device), target.to(device).long().squeeze()
            output = model(data)
            target = target.argmax(dim=0, keepdim=True)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            loss = ce(output, target)
            test_loss += loss.item()
            correct = (predictions == target).sum().item()
            test_acc += correct / 1
            
            tepoch.set_postfix(loss=test_loss/i, accuracy=test_acc/i)
            sleep(0.1)