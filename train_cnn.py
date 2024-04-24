from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore") # Ignore warnings
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim
from torchvision import transforms as T
from transformers import ResNetModel,ResNetConfig,MobileNetV2Config, MobileNetV2Model,MobileNetV1Config, MobileNetV1Model
import timm

from sklearn.metrics import classification_report

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # Some images are truncated

class YogaPoseDataset(Dataset):
    def __init__(self,poses_to_path,poses_to_idx,poses,transform=None):
        self.poses_to_path = poses_to_path
        self.max_size = max(len(i) for i in poses_to_path)
        self.poses_to_idx = poses_to_idx
        self.class_num = len(self.poses_to_idx)
        self.poses=poses
        self.transform=transform
    def __len__(self):
        return self.max_size*self.class_num
    def __getitem__(self,idx):
        pose_class = idx%self.class_num
        img_id = (idx//self.class_num)%len(self.poses_to_path[self.poses[pose_class]])
        img = Image.open(self.poses_to_path[self.poses[pose_class]][img_id]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img,pose_class

# Model Configurations
def get_model(model_name,num_class,pretrained=True,dropout=0.0):
    # CNN Backbone
    if model_name == "resnet-18":
        if pretrained:
            model = ResNetModel.from_pretrained("microsoft/resnet-18")
        else:
            model = ResNetModel(ResNetConfig(depths=[2,2,2,2]))
    elif model_name == "resnet-34":
        if pretrained:
            model = ResNetModel.from_pretrained("microsoft/resnet-34")
        else:
            model = ResNetModel(ResNetConfig(depths=[3,4,6,3]))
    elif model_name == "resnet-50":
        if pretrained:
            model = ResNetModel.from_pretrained("microsoft/resnet-50")
        else:
            model = ResNetModel(ResNetConfig(depths=[3,4,6,3]))
    elif model_name == "mobilenetv2":
        if pretrained:
            model = MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.4_224")
        else:
            model = MobileNetV2Model(MobileNetV2Config())
    elif model_name == "mobilenetv1":
        if pretrained:
            model = MobileNetV1Model.from_pretrained("google/mobilenet_v1_1.0_224")
        else:
            model = MobileNetV1Model(MobileNetV1Config())
    elif model_name == "mobilenetv3":
        model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained)
    # MLP Head
    if model_name.startswith("resnet"):
        linear_proj = nn.Sequential(
            nn.BatchNorm1d(model.config.hidden_sizes[-1]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(model.config.hidden_sizes[-1],256)
        )
    elif model_name=="mobilenetv2":
        linear_proj = nn.Sequential(
            nn.BatchNorm1d(1280),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(1280,256)
        )
    elif model_name=="mobilenetv1":
        linear_proj = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024,256)
        )
    elif model_name=="mobilenetv3":
        linear_proj = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(1000,256),
        )
    # Final MLP Head
    mlp = nn.Sequential(
        linear_proj,
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256,num_class)
    )
    return model,mlp
        

def parse():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--datadir', type=str, default="yoga-82")
    parser.add_argument('--model', type=str, default="mobilenetv3", \
                        choices=["resnet-18","resnet-34","resnet-50","resnet-101","resnet-152","mobilenetv2","mobilenetv1","mobilenetv3"])
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--pretrained', type=bool, default=True)
    args = parser.parse_args()
    return args

def main():
    print("Is cuda available: ", torch.cuda.is_available())

    args = parse()
    poses = []
    poses_to_path_train = {}
    poses_to_path_test = {}
    poses_to_path_val = {}
    for dirname, _, filenames in os.walk(os.path.join(args.datadir,'train/')):
        _dirname = dirname.split('/')[-1].lower()
        poses.append(_dirname)
        poses_to_path_train[_dirname] = []
        for filename in filenames:
            poses_to_path_train[_dirname].append(os.path.join(dirname, filename))
    for dirname, _, filenames in os.walk(os.path.join(args.datadir,'test/')):
        _dirname = dirname.split('/')[-1].lower()
        poses_to_path_test[_dirname] = []
        for filename in filenames:
            poses_to_path_test[_dirname].append(os.path.join(dirname, filename))

    for dirname, _, filenames in os.walk(os.path.join(args.datadir,'valid/')):
        _dirname = dirname.split('/')[-1].lower()
        poses_to_path_val[_dirname] = []
        for filename in filenames:
            poses_to_path_val[_dirname].append(os.path.join(dirname, filename))

    # Clean the parent directory
    poses=[i.split('/')[-1].lower() for i in poses[1:]]
    del poses_to_path_train['']
    del poses_to_path_test['']
    del poses_to_path_val['']


    poses_to_idx = dict((j,i) for i,j in enumerate(poses))

    # Data augmentations
    transform = T.Compose([
        T.Resize((256,256), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
        T.RandomErasing(p=0.5,value="random"),
        T.ColorJitter(),
        T.RandomRotation((-45,45)),
        T.RandomHorizontalFlip(p=0.5)
    ]
    )

    val_trans = T.Compose([
            T.Resize((256,256), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
    ]
    )
    train_ds = YogaPoseDataset(poses_to_path_train,poses_to_idx,poses,transform)
    val_ds = YogaPoseDataset(poses_to_path_val,poses_to_idx,poses,val_trans)


    model,mlp = get_model(args.model,len(poses_to_idx),args.pretrained,args.dropout)

    opt = torch.optim.AdamW([*model.parameters(),*mlp.parameters()],lr=args.lr,weight_decay=args.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    BATCH_SIZE = args.batch_size
    train_dl = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    val_dl  = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    # region Train
    model.to(device)
    mlp.to(device)
    for epoch in range(args.epochs):
        model.train()
        mlp.train()
        tr_loss = []
        res = []
        for step, batch in enumerate(train_dl):
            imgs,labels = batch
            encodings = model(imgs.to(device))
            if args.model=="mobilenetv3":
                logits = mlp(encodings)
            else:
                logits = mlp(encodings.pooler_output.flatten(1))
            loss = nn.functional.cross_entropy(logits,labels.to(device))
            tr_loss.append(loss.detach().cpu())
            # Training accuracy
            res.append(torch.argmax(logits,dim=1).detach()==labels.to(device))
            loss.backward()
            opt.step()
            opt.zero_grad()
        tr_loss = torch.stack(tr_loss).float().mean() 
        tr_acc = torch.cat([i for i in res]).float().mean()
        model.eval()
        mlp.eval()
        res = []
        val_loss = []
        for step, batch in enumerate(val_dl):
            with torch.no_grad():
                imgs,labels = batch
                encodings = model(imgs.to(device))
                if args.model=="mobilenetv3":
                    logits = mlp(encodings)
                else:
                    logits = mlp(encodings.pooler_output.flatten(1))
                loss = nn.functional.cross_entropy(logits,labels.to(device))
                # Validation accuracy
                res.append(torch.argmax(logits,dim=1)==labels.to(device))
                val_loss.append(loss.detach())
        print(f"{epoch} Tr loss: {tr_loss} Tr acc: {tr_acc}","Val loss: ",torch.stack(val_loss).float().mean(),"Val acc: ",torch.cat([i for i in res]).float().mean())

    # endregion

    # region Test
    test_ds = YogaPoseDataset(poses_to_path_test,poses_to_idx,poses,val_trans)
    test_dl  = DataLoader(test_ds,batch_size=BATCH_SIZE,num_workers=4)
    model.eval()
    mlp.eval()
    res = []
    test_loss = []
    test_scores = []
    test_labels = []
    for step, batch in enumerate(test_dl):
        with torch.no_grad():
            imgs,labels = batch
            encodings = model(imgs.to(device))
            if args.model=="mobilenetv3":
                logits = mlp(encodings)
            else:
                logits = mlp(encodings.pooler_output.flatten(1))
            test_scores.append(torch.argmax(logits,dim=1))
            test_labels.append(labels)
            loss = nn.functional.cross_entropy(logits,labels.to(device))
            res.append(torch.argmax(logits,dim=1)==labels.to(device))
            test_loss.append(loss.detach())
    print("Test loss: ",torch.stack(test_loss).float().mean(),"Test acc: ",torch.cat([i for i in res]).float().mean())

    # Other metrics
    test_scores = torch.cat(test_scores).cpu().numpy()
    test_labels = torch.cat(test_labels).cpu().numpy()
    print(classification_report(test_labels,test_scores))

    # endregion
    
    # Save the model
    torch.save(model.state_dict(),f"saved_cnn_models/{args.model}_model.pth")
    torch.save(mlp.state_dict(),f"saved_cnn_models/{args.model}_mlp.pth")
    return 

if __name__ == "__main__":
    main()