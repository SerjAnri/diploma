import torch
from sklearn import metrics


def fit_one_cycle(epochs, model, train_loader, optimizer):
    model.train()  # Устанавливаем режим обучения
    total_loss = 0

    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            # Перемещение на GPU, если доступно
            images, labels = images.cuda(), labels.cuda() if torch.cuda.is_available() else (images, labels)

            # Обнуляем градиенты
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item() / epochs / len(train_loader)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}')
    print(f'Total Loss: {total_loss}')

def validate(model, val_loader, device):
    model.eval()

    gt_labels = []
    pred_labels = []
    pred_scores = []

    for img, label in val_loader:
        img = img.to(device)

        with torch.no_grad():
            out = model(img)

        for j in range(out.size(0)):
            pred_scores.append(out[j].to("cpu").numpy())

        _, predicted = torch.max(out.data, 1)

        gt_labels += label.to('cpu').numpy().tolist()
        pred_labels += predicted.to("cpu").numpy().tolist()

    acc = metrics.accuracy_score(gt_labels, pred_labels)
    f1_macro = metrics.f1_score(gt_labels, pred_labels, average="macro")
    top5acc = metrics.top_k_accuracy_score(gt_labels, pred_scores, k=5, labels=list(range(10)))

    print("Model accuracy: {:.5f}".format(acc))
    print("Model F1 macro: {:.5f}".format(f1_macro))
    print("Model Top5Accuracy: {:.5f}".format(top5acc))
    return dict(accuracy=acc, f1_macro=f1_macro, top5acc=top5acc)