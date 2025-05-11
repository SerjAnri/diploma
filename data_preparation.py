from mpmath.identification import transforms
import os
import pandas as pd
from torchvision import datasets, transforms

# Директории для сохранения изображений
train_dir = './data/train/train'
test_dir = './data/test/test'

# Файлы для сохранения меток
train_labels_csv = 'train_labels.csv'
test_labels_csv = 'test_labels.csv'

# Создаём директории, если они не существуют
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразуем в тензор
])


# Загружаем CIFAR-10
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Функция для сохранения изображений и меток
def save_images_and_labels(dataset, save_dir, label_file):
    labels_data = []
    for i, (image, label) in enumerate(dataset):
        image = transforms.ToPILImage()(image)
        # Сохраняем изображение с пронумерованным названием
        image.save(os.path.join(save_dir, f'{i}.png'))  # Форматируем номер с предшествующими нулями
        labels_data.append({'id': i, 'label': label})  # Собираем метки
    return labels_data

# Сохраняем метки и изображения для обучающего набора
train_labels = save_images_and_labels(trainset, train_dir, train_labels_csv)

# Сохраняем метки и изображения для тестового набора
test_labels = save_images_and_labels(testset, test_dir, test_labels_csv)

# Создаём DataFrame для обучающих меток и сохраняем в CSV
train_labels_df = pd.DataFrame(train_labels)
train_labels_df.to_csv(train_labels_csv, index=False)

# Создаём DataFrame для тестовых меток и сохраняем в CSV
test_labels_df = pd.DataFrame(test_labels)
test_labels_df.to_csv(test_labels_csv, index=False)

print(f'Images and labels saved:\nTraining images in {train_dir} and labels in {train_labels_csv}\nTesting images in {test_dir} and labels in {test_labels_csv}')