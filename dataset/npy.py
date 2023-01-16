import numpy as np
import pandas as pd




data1 = np.load("C:/Users/PC03/Desktop/dance_learning/dataset/raw_dance_1673845565.npy") # 데이터 로드. @파일명

df = pd.DataFrame(data1)
df = df.T
df.to_csv('sample.csv', index=False)