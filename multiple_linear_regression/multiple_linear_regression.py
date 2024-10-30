# in simple linear regression y=b0+b1*x
#maas=b0+b1*deneyim
# in multiple linear regression
#y=b0+b1*x1+b2*x2

#mass yani y değeri aslında burda dependent variable olarak geçmektedir.
#deneyim ve yas değiskenlerimzi de independent variableo larak geçmektedir.
#b0,b1,b2
# AMACIMIZ MEAN SQUARE ERROR'umuzu minimuma düşürmek

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Veri yükleme
df = pd.read_csv('../multiple_linear_regression_dataset.csv', sep=";")
x = df.iloc[:, [0, 2]].values  # 'deneyim' ve 'yas' sütunları
y = df['maas'].values.reshape(-1, 1)

# Model oluşturma
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x, y)

# Sonuçları yazdırma
print("b0:", multiple_linear_regression.intercept_)
print("b1, b2:", multiple_linear_regression.coef_)
print(multiple_linear_regression.predict(np.array([[10, 35], [5, 35]])))

# Klasör yolunu belirleme ve oluşturma
output_dir = "images/multiple_linear_regression"
os.makedirs(output_dir, exist_ok=True)

# Deneyime göre maaşı görselleştirme ve kaydetme
plt.scatter(df['deneyim'], y, color='blue', label="Gerçek Maaş")
plt.plot(df['deneyim'], multiple_linear_regression.predict(x), color='red', label="Tahmin Edilen Maaş")
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.legend()
plt.title("Deneyime Göre Maaş Tahmini")
plt.savefig(os.path.join(output_dir, "experience_vs_salary.png"))
plt.close()  # Grafiği kapatma

# Yaşa göre maaşı görselleştirme ve kaydetme
plt.scatter(df['yas'], y, color='blue', label="Gerçek Maaş")
plt.plot(df['yas'], multiple_linear_regression.predict(x), color='red', label="Tahmin Edilen Maaş")
plt.xlabel("Yaş")
plt.ylabel("Maaş")
plt.legend()
plt.title("Yaşa Göre Maaş Tahmini")
plt.savefig(os.path.join(output_dir, "age_vs_salary.png"))
plt.close()  # Grafiği kapatma
