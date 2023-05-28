import os
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def main():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'D:\\Downloads\\Data\\Corn_Split'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=36, shuffle=True, num_workers=4) for x in
                   ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        best_model_wts = model.state_dict()
        best_acc = 0.0
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'train':
                    train_losses.append(epoch_loss)
                else:
                    val_losses.append(epoch_loss)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

            print()

        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)

        # Plot loss vs. epochs
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('Adam_loss_vs_epochs.png')

        return model, train_losses, val_losses

    model_ft = models.resnet18(pretrained=False)
    model_ft.load_state_dict(torch.load('D:\\Downloads\\\Data\\resnet18-f37072fd.pth'))
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=0.0001)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.5)

    # 训练和评估模型的部分
    trained_model, train_losses, val_losses = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                                          num_epochs=100)

    # 评估模型
    trained_model.eval()

    running_corrects = 0
    true_labels = []
    predicted_labels = []

    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = trained_model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data).item()
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())

    accuracy = running_corrects / dataset_sizes['val']
    print('Accuracy: {:.4f}'.format(accuracy))

    report = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.axis('off')
    ax.table(cellText=report_df.values, colLabels=report_df.columns, rowLabels=report_df.index, cellLoc='center',
             loc='center')
    plt.savefig("Adam_classification_report.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    cm = confusion_matrix(true_labels, predicted_labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.set_style("white")
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=class_names, yticklabels=class_names, cbar=False,
                annot_kws={"size": 16})

    # 设置轴标签的颜色
    for tick_label in ax.get_xticklabels():
        tick_label.set_color('red')  # 设置颜色为红色
    for tick_label in ax.get_yticklabels():
        tick_label.set_color('red')  # 设置颜色为红色

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("Adam_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
