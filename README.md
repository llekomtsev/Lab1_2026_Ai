
# Лабораторная работа 1  
## Классификация пород собак Stanford Dogs с использованием SqueezeNet

**Дисциплина:** Проектирование систем интеллектуального анализа промышленных данных  
**Фреймворк:** PyTorch  
**Среда выполнения:** Google Colab / Kaggle GPU  

---

## 1. Цель работы

Цель лабораторной работы — реализовать алгоритм глубокого обучения для задачи классификации изображений, провести обучение нейронной сети в нескольких режимах и сравнить качество полученных моделей.

В работе исследуются:

- дообучение предобученной модели;
- обучение модели с нуля;
- сравнение оптимизаторов Adam и AmsGrad;
- влияние transfer learning на качество классификации.

---

## 2. Задание

Необходимо:

1. Скачать **Stanford Dogs Dataset**.
2. Разделить датасет на обучающую, валидационную и тестовую выборки в соотношении **70 / 15 / 15**.
3. Реализовать нейронную сеть с использованием **PyTorch**.
4. Провести три эксперимента:
   - предобученная модель + Adam;
   - предобученная модель + оптимизатор по варианту;
   - модель с нуля + Adam.
5. Сравнить качество моделей по метрикам:
   - Precision;
   - Recall;
   - F1-score.

---

## 3. Вариант

| Компонент | Значение |
|---|---|
| Архитектура | SqueezeNet |
| Аугментация | Gaussian Blur |
| Оптимизатор | AmsGrad |

---

## 4. Датасет

В работе использовался **Stanford Dogs Dataset**.

Датасет содержит изображения собак, распределённые по **120 породам**.

После загрузки датасета было получено:

| Параметр | Значение |
|---|---:|
| Количество классов | 120 |
| Общее количество изображений | 20 580 |

Разделение датасета:

| Выборка | Доля | Количество изображений |
|---|---:|---:|
| Train | 70% | 14 405 |
| Validation | 15% | 3 087 |
| Test | 15% | 3 088 |

---

## 5. Предобработка данных

Для обучающей выборки использовались следующие преобразования:

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])
````

Для валидационной и тестовой выборок использовались только базовые преобразования:

```python
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])
```

Аугментация **Gaussian Blur** была использована в соответствии с вариантом лабораторной работы. Остальные преобразования применялись как стандартные аугментации изображений для повышения устойчивости модели к изменениям положения объекта, освещения и поворота.

---

## 6. Архитектура модели

В работе использовалась архитектура **SqueezeNet 1.1**.

SqueezeNet — это компактная свёрточная нейронная сеть для классификации изображений. Она использует специальные **Fire-модули**, которые позволяют уменьшить количество параметров модели и при этом сохранить способность извлекать признаки из изображений.

### 6.1. Fire-модуль

В рамках лабораторной работы была реализована собственная версия Fire-модуля.

Fire-модуль состоит из трёх основных свёрточных частей:

* `squeeze` — свёртка `1×1`, уменьшающая количество каналов;
* `expand1x1` — свёртка `1×1`;
* `expand3x3` — свёртка `3×3`.

После выполнения двух expand-веток их выходы объединяются по каналам с помощью `torch.cat`.

```python
class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModule, self).__init__()

        self.squeeze = nn.Conv2d(
            in_channels=in_channels,
            out_channels=squeeze_channels,
            kernel_size=1
        )

        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(
            in_channels=squeeze_channels,
            out_channels=expand1x1_channels,
            kernel_size=1
        )

        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(
            in_channels=squeeze_channels,
            out_channels=expand3x3_channels,
            kernel_size=3,
            padding=1
        )

        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))

        out1 = self.expand1x1_activation(self.expand1x1(x))
        out2 = self.expand3x3_activation(self.expand3x3(x))

        return torch.cat([out1, out2], dim=1)
```

### 6.2. Собственная реализация SqueezeNet

Для эксперимента обучения с нуля была самостоятельно реализована архитектура **CustomSqueezeNet**, соответствующая принципам SqueezeNet 1.1.

Общая структура модели:

1. начальная свёртка `Conv2d`;
2. `ReLU`;
3. `MaxPool2d`;
4. последовательность Fire-модулей;
5. `Dropout`;
6. финальная свёртка `Conv2d(512, 120, kernel_size=1)`;
7. `AdaptiveAvgPool2d`.

```python
class CustomSqueezeNet(nn.Module):
    def __init__(self, num_classes=120):
        super(CustomSqueezeNet, self).__init__()

        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireModule(64, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireModule(128, 32, 128, 128),
            FireModule(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            FireModule(512, 64, 256, 256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x
```

В экспериментах с предобученной моделью использовалась стандартная реализация **SqueezeNet 1.1** из `torchvision` с весами ImageNet, так как по заданию требовалось использовать предобученную модель.

Изначально предобученная SqueezeNet рассчитана на классификацию **1000 классов ImageNet**. В данной работе требуется классифицировать **120 пород собак**, поэтому последний классификационный слой был заменён:

```python
model.classifier[1] = nn.Conv2d(
    512,
    NUM_CLASSES,
    kernel_size=(1, 1)
)

model.num_classes = NUM_CLASSES
```

где:

```python
NUM_CLASSES = 120
```

Таким образом, модель была адаптирована под задачу классификации 120 пород собак.

---

## 7. Параметры обучения

| Параметр           |                  Значение |
| ------------------ | ------------------------: |
| Размер изображения |                 224 × 224 |
| Batch size         |                        64 |
| Количество эпох    |                        15 |
| Функция потерь     |          CrossEntropyLoss |
| Framework          |                   PyTorch |
| Среда выполнения   | Google Colab / Kaggle GPU |

---

## 8. Эксперименты

В работе были проведены три эксперимента.

### 8.1. Эксперимент 1 — Pretrained SqueezeNet + Adam

Использовалась стандартная модель **SqueezeNet 1.1** из `torchvision` с предобученными весами ImageNet.

Оптимизатор:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    amsgrad=False
)
```

---

### 8.2. Эксперимент 2 — Pretrained SqueezeNet + AmsGrad

Использовалась стандартная модель **SqueezeNet 1.1** из `torchvision` с предобученными весами ImageNet.

AmsGrad был включён через параметр `amsgrad=True` в оптимизаторе Adam:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    amsgrad=True
)
```

---

### 8.3. Эксперимент 3 — Custom SqueezeNet From Scratch + Adam

Для обучения с нуля использовалась собственная реализация архитектуры **SqueezeNet**, написанная на PyTorch через классы `FireModule` и `CustomSqueezeNet`.

Модель создавалась без предобученных весов:

```python
model = CustomSqueezeNet(num_classes=120)
```

Оптимизатор:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
    amsgrad=False
)
```

---

## 9. Метрики качества

Для оценки качества моделей использовались метрики **Precision**, **Recall** и **F1-score**.

| Метрика   | Описание                                                                                             |
| --------- | ---------------------------------------------------------------------------------------------------- |
| Precision | Показывает, какая доля объектов, отнесённых моделью к классу, действительно принадлежит этому классу |
| Recall    | Показывает, какую долю объектов данного класса модель смогла правильно найти                         |
| F1-score  | Объединяет Precision и Recall в одну метрику                                                         |

Так как задача является многоклассовой, метрики рассчитывались с использованием усреднения:

```python
average='weighted'
```

---

## 10. Результаты

Итоговые результаты на тестовой выборке:

| Эксперимент                      | Precision | Recall | F1-score |
| -------------------------------- | --------: | -----: | -------: |
| Pretrained SqueezeNet + Adam     |    0.5811 | 0.5253 |   0.5242 |
| Pretrained SqueezeNet + AmsGrad  |    0.5804 | 0.5525 |   0.5458 |
| Custom SqueezeNet Scratch + Adam |    0.0525 | 0.0606 |   0.0437 |

Лучшее значение F1-score показала модель:

> **Pretrained SqueezeNet + AmsGrad**

---

## 11. Графики обучения

### 11.1. Pretrained SqueezeNet + Adam

<img width="567" height="455" alt="Experiment 1: Pretrained SqueezeNet + Adam" src="https://github.com/user-attachments/assets/7a0524e4-23ce-46e3-b873-b76703a5f905" />

График потерь Pretrained SqueezeNet + Adam.

---

### 11.2. Pretrained SqueezeNet + AmsGrad

<img width="567" height="455" alt="Experiment 2: Pretrained SqueezeNet + AmsGrad" src="https://github.com/user-attachments/assets/d4c9dd38-ef0a-472e-adea-6d36fa6707a3" />

График потерь Pretrained SqueezeNet + AmsGrad.

---

### 11.3. Custom SqueezeNet From Scratch + Adam

<img width="584" height="455" alt="Experiment 3: Custom SqueezeNet From Scratch + Adam" src="https://github.com/user-attachments/assets/7dec6580-3c33-4186-85f3-85bbb3b295bc" />

График потерь Custom SqueezeNet From Scratch + Adam.

---

## 12. Анализ результатов

### 12.1. Сравнение Adam и AmsGrad

Результаты предобученных моделей:

| Модель                          | F1-score |
| ------------------------------- | -------: |
| Pretrained SqueezeNet + Adam    |   0.5242 |
| Pretrained SqueezeNet + AmsGrad |   0.5458 |

Разница между AmsGrad и Adam:

```text
0.5458 - 0.5242 = 0.0216
```

AmsGrad показал результат выше примерно на **0.0216** по F1-score.

Это означает, что в данном эксперименте AmsGrad оказался немного эффективнее обычного Adam.

---

### 12.2. Сравнение предобученной модели и модели с нуля

| Модель                           | F1-score |
| -------------------------------- | -------: |
| Pretrained SqueezeNet + AmsGrad  |   0.5458 |
| Custom SqueezeNet Scratch + Adam |   0.0437 |

Модель, обученная с нуля на собственной реализации SqueezeNet, показала значительно более низкое качество по сравнению с предобученными моделями.

При этом значение F1-score для модели с нуля оказалось выше случайного угадывания для 120 классов:

```text
1 / 120 ≈ 0.0083
```

Это показывает, что собственная реализация SqueezeNet начала обучаться, однако 15 эпох недостаточно для достижения качества предобученной модели.

Причины более низкого качества модели с нуля:

* модель не использовала предобученные признаки ImageNet;
* датасет содержит 120 классов пород собак;
* классы визуально похожи между собой;
* обучение с нуля требует большего количества эпох;
* для улучшения результата требуется более тщательный подбор learning rate и scheduler.

---

## 13. Выводы

В ходе лабораторной работы была реализована классификация пород собак на датасете **Stanford Dogs Dataset** с использованием архитектуры **SqueezeNet**.

Для экспериментов с предобучением использовалась стандартная SqueezeNet 1.1 с весами ImageNet. Для эксперимента обучения с нуля была самостоятельно реализована архитектура SqueezeNet через Fire-модули.

Были проведены три эксперимента:

1. **Pretrained SqueezeNet + Adam**
2. **Pretrained SqueezeNet + AmsGrad**
3. **Custom SqueezeNet Scratch + Adam**

По результатам экспериментов можно сделать следующие выводы:

* использование предобученной модели значительно повышает качество классификации;
* лучший результат показала модель **Pretrained SqueezeNet + AmsGrad**;
* AmsGrad оказался немного эффективнее обычного Adam в данном эксперименте;
* собственная реализация SqueezeNet с нуля начала обучаться, но значительно уступила предобученным моделям;
* для обучения модели с нуля требуется больше эпох, подбор learning rate и использование scheduler;
* transfer learning является наиболее эффективным подходом для данной задачи при ограниченном числе эпох и ограниченных вычислительных ресурсах.

---

## 14. Использованные источники

1. [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
2. [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters](https://arxiv.org/abs/1602.07360)
3. [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
4. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
5. [Torchvision Models Documentation](https://pytorch.org/vision/stable/models.html)

---

## 15. Код

Исходный код находится в файле:

```text
lab1_stanford_dogs.ipynb
```



В репозитории также размещены:

* графики обучения;
* сохранённые веса лучших моделей;
* итоговые результаты экспериментов.

```

Единственное: ссылки на картинки в разделе графиков у тебя пока старые. Надо заменить `src="..."` на новые GitHub-ссылки после того, как загрузишь свежие графики с новыми результатами.
```
