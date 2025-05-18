import torch
from sklearn import metrics


def validate(model, val_loader, device, is_threshold = False):
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
    if is_threshold:
        print("Threshold model accuracy: {:.5f}".format(acc))
        print("Threshold model F1 macro: {:.5f}".format(f1_macro))
        print("Threshold model Top5Accuracy: {:.5f}".format(top5acc))
    else:
        print("Final model accuracy: {:.5f}".format(acc))
        print("Final model F1 macro: {:.5f}".format(f1_macro))
        print("Final model Top5Accuracy: {:.5f}".format(top5acc))
    return acc