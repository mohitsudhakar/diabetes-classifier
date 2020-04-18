import torch
from torch.nn.modules.loss import BCELoss
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader

from dataloader import DiabetesDataset
from model import Model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DiabetesDataset('diabetes.csv')

    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

    model = Model()

    optim = SGD(model.parameters(), lr=0.1)
    num_epochs = 5

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs.to(device)
            labels.to(device)

            y_pred = model(inputs)
            y_pred = y_pred.squeeze(dim=1)
            print("Y_pred", y_pred)
            print("Y_true", labels)

            loss = BCELoss(y_pred, labels)
            print(epoch, i, loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()
