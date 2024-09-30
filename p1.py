import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_true = []
y_pred = []

sciezka = '/home/marcin/Pulpit/Dataset/dataset_kopia/kopia_Test'

PP = 0
PN = 0
FP = 0
FN = 0

for kategoria in os.listdir(sciezka):
    sciezka_folder = os.path.join(sciezka, kategoria)
    if os.path.isdir(sciezka_folder):
        for plik in os.listdir(sciezka_folder):
            if plik.endswith(('.jpg', '.jpeg', '.png')):
                sciezka_zdj = os.path.join(sciezka_folder, plik)
                zdj = cv2.imread(sciezka_zdj)

                if zdj is None:
                    print(f"Nie można odczytać pliku: {sciezka_zdj}")
                    continue

                szarosc = cv2.cvtColor(zdj, cv2.IMREAD_GRAYSCALE)

                # Zastosowanie filtru Sobela
                # sobel_x = cv2.Sobel(zdj, cv2.CV_64F, 1, 0, ksize=3)
                # sobel_y = cv2.Sobel(zdj, cv2.CV_64F, 0, 1, ksize=3)
                # magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                #
                # threshold = 50
                # krawedzie = magnitude > threshold
                #licz_krawedzie = np.sum(krawedzie)

                # Detekcja krawędzi Canny'ego
                krawedzie = cv2.Canny(szarosc, 150, 250)
                licz_krawedzie = np.sum(krawedzie>0)

                if licz_krawedzie < 5000:
                    przewidywana_kategoria = 'slaba'
                else:
                    przewidywana_kategoria = 'dobra'

                prawdziwa_kategoria = 'slaba' if 'very_low' in kategoria else 'dobra'

                y_true.append(prawdziwa_kategoria)
                y_pred.append(przewidywana_kategoria)

                # Klasyfikacja
                if przewidywana_kategoria == prawdziwa_kategoria:
                    if przewidywana_kategoria == 'slaba':
                        PP += 1
                    else:
                        PN += 1
                else:
                    if przewidywana_kategoria == 'dobra':
                        FN += 1
                    else:
                        FP += 1

# Tworzenie wykresu (nazwy angielskie, polskie nie zmiescily
labels = ['PP', 'PN', 'FP', 'FN']
values = [PP, FN, FP, PN]

x = np.arange(len(labels))

plt.bar(x, values, color=['g', 'b', 'r', 'orange'])
plt.xticks(x, labels)
plt.ylabel('Liczba próbek')
plt.title('Wyniki klasyfikacji')
plt.show()

# Tworzenie tabeli wyników
results = {
    'Kategoria': labels,
    'Liczba próbek': values
}

# Macierz błędów
cm = confusion_matrix(y_true, y_pred, labels=['slaba', 'dobra'])

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Słaba widoczność', 'Dobra widoczność'], yticklabels=['Słaba widoczność', 'Dobra widoczność'])
plt.xlabel('Predykcja')
plt.ylabel('Stan faktyczny')
plt.title('Macierz błędów dla danych testowych P1')
plt.show()
