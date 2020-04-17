from torch.autograd import Variable
from torch.nn.modules.loss import BCELoss
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader

from dataloader import DiabetesDataset
from model import Model


if __name__ == '__main__':
    dataset = DiabetesDataset('diabetes.csv')

    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

    model = Model()

    optim = SGD(model.parameters(), lr=0.1)
    num_epochs = 5

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            y_pred = model(inputs)

            loss = BCELoss(y_pred, labels)
            print(epoch, i, loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()
