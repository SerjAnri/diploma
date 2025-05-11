import torch
from torch import optim


def fit_one_cycle(epochs, model, train_loader):
    model.train()  # Устанавливаем режим обучения
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            # Перемещение на GPU, если доступно
            images, labels = images.cuda(), labels.cuda() if torch.cuda.is_available() else (images, labels)

            # Обнуляем градиенты
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}')

def validate(model, val_loader):
    model.eval()  # Устанавливаем режим оценки
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda() if torch.cuda.is_available() else (images, labels)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    result = correct / total
    print(f'Accuracy: {100 * result}%')
    return result