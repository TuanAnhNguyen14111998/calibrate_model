import torch
from torchsummary import summary
from torch.utils import data
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import albumentations as A
import pkbar
import sys
import os
path = os.path.dirname(__file__)
root_folder = os.path.join(
    os.path.abspath(path).split("calibrate_model")[0],
    "calibrate_model"
)
sys.path.insert(0, root_folder)

from src.base_line.efficienet import Net
from src.base_line.data_loader import Dataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class Training():
    def __init__(self, train_csv, test_csv, class_names):
        self.df_train = pd.read_csv(train_csv)
        self.df_test = pd.read_csv(test_csv)
        self.class_names = class_names
        self.dataframe_filter()

    def dataframe_filter(self):
        self.df_train = self.df_train[
            self.df_train.class_name.isin(self.class_names)
        ]

        self.df_test = self.df_test[
            self.df_test.class_name.isin(self.class_names)
        ]
    
    def convert_label(self, class_names):
        dict_label = {}
        for index, class_name in enumerate(class_names):
            dict_label[class_name] = index
        
        return dict_label
    
    def get_loss(self, outputs, labels):
        loss = nn.CrossEntropyLoss()(outputs, labels)

        return loss
    
    def get_optimizer(self, model, lr=0.001, momentum=0.9):
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        return optimizer

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        for param in model.backbone._conv_head.parameters():
            param.requires_grad = True
        for param in model.backbone._fc.parameters():
            param.requires_grad = True
        
        return model
    
    def training_model(self, model_name="efficientnet-b0",
                        image_size=(224, 224), num_epochs=10, batch_size=10,
                        learning_rate=0.01, momentum=0.9):
        labels = self.convert_label(self.class_names)
        
        params = {
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 6
        }

        transform_train = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, 
                rotate_limit=15, p=0.5
            ),
            A.RGBShift(
                r_shift_limit=15, g_shift_limit=15, 
                b_shift_limit=15, p=0.5
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            )
        ])

        transform_test = A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            )
        ])

        training_set = Dataset(self.df_train, labels, image_size, transform_train)
        training_generator = data.DataLoader(training_set, **params)

        test_set = Dataset(self.df_test, labels, image_size, transform_test)
        test_generator = data.DataLoader(test_set, **params)

        net = Net(model_name=model_name, n_class=len(self.class_names))
        net = net.to(device)

        train_params = [param for param in net.parameters() if param.requires_grad]

        optimizer = self.get_optimizer(net, lr=learning_rate, momentum=momentum)

        train_per_epoch = len(training_generator)
        for epoch in range(num_epochs):
            train_loss = []
            kbar = pkbar.Kbar(
                target=train_per_epoch, epoch=epoch, 
                num_epochs=num_epochs, width=8, 
                always_stateful=False
            )
            for index, local_data in enumerate(training_generator):
                local_images = local_data[0].to(device)
                local_labels = local_data[1].to(device)

                optimizer.zero_grad()

                outputs = net(local_images)
                _, predicted = torch.max(outputs.data, 1)
                
                loss = self.get_loss(outputs, local_labels)
                
                loss.backward()
                optimizer.step()

                kbar.update(index, values=[("train_loss", loss)])

                train_loss.extend([loss.item()])
            
            train_loss = np.mean(np.array(train_loss))

            total = 0
            correct = 0
            test_losses = []
            with torch.no_grad():
                for index, local_data in enumerate(test_generator):
                    local_images = local_data[0].to(device)
                    local_labels = local_data[1].to(device)
                    
                    outputs = net(local_images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += local_labels.size(0)
                    correct += (predicted == local_labels).sum().item()
                    loss = self.get_loss(outputs, local_labels)

                    test_losses.extend([loss.item()])
                
                test_losses = np.mean(np.array(val_losses))
            
            test_acc = (100 * correct / total)
            kbar.update(1, values=[
                ("train_loss", train_loss), 
                ("test_losses", test_losses), 
                ("test_acc", test_losses)
            ])
            





if __name__ == "__main__":
    train_csv = "./datasets/clipart_info/train.csv"
    test_csv = "./datasets/clipart_info/test.csv"
    class_names = [
        "swan", "dumbbell", "stairs" "suitcase", "coffee_cup",
        "strawberry", "submarine", "stethoscope", "whale", "bird",
        "streetlight", "zigzag", "tiger",
    ]
    training = Training(
        train_csv=train_csv, 
        test_csv=test_csv, 
        class_names=class_names
    )

    training.training_model(
        model_name="efficientnet-b0",
        image_size=(225, 225),
        num_epochs=2,
        batch_size=1
    )
