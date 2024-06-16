Данные библиотеки необходимы для работы программы

```b
pip install pytorch-lifestream
pip install polars==0.20.31
pip install pytorch_lightning==1.9.0
pip install pandas==2.0.3
pip isntall numpy==1.24.2
```
Файл train.ipynb содержит код для обучения модели, inference.ipynb - для получения предиктов. Для корректной работы необходимо прописать пути к данным.

Так как я работал в Google Colab, то путь к файлам у меня был в формате '/content/Hackathon/'. Это можно исправить, поменяв в соответсвующих ячейках пути (4 ячейка).
