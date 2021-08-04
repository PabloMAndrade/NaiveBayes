# P(Alto) = 6/14 * 1/6 * 4/6 * 6/6 * 1/6 = 0,0079 
# P(Moderado) = 3/14 * 1/3 * 1/3 * 2/3 * 1/3 = 0,0052
# P(Baixo) = 5/14 * 3/5 * 2/5 * 3/5 * 5/5 = 0,0514
# Quem ganhou foi o risco baixo , pois é o mais "grande" em relação aos outros , do cliente dar um golpe no banco.

# Soma = 0,0079 + 0,0052 + 0,0514 = 0,0645 equivale a 100% pois é a soma de todos
# Precisa do valor da soma pra fazer a probabilidade

# P(Alto) = 0,0079 / 0,645 * 100 = 12,24%
# P(Moderado) = 0,0052 / 0,645 * 100 = 8,06%
# P(Baixo) = 0,0514 / 0,645 * 100 = 79,68%

# Outro exemplo: Historia Ruim , Divida Alta , Garantias Adequadas , Renda < 15
# P(Alto) = 6/14 * 3/6 * 4/6 * 0 * 3/6 = 0
# P(Moderado) = 3/14 * 1/3 * 1/3 * 1/3 * 0 = 0
# P(Baixo) = 5/14 * 0 * 2/5 * 2/5 * 0 = 0

# Correção laplaciana (quando tem 0 na conta , todo número multiplicado por 0 é 0) ela adicionará um registro adicional
# Adicionamos 1 na fração , exemplo : Antes alto era 6/14 , agora é 7/15 , Moderado era 3/14 agora é 4/15 , Baixo : 5/14 e agora : 8/17

# P(Alto) = 7/15 * 3/6 * 4/6 * 1/6 * 3/6 = 0,54
# P(Moderado) = 4/15 * 1/3 * 1/3 * 1/3 * 1/3 = 0.0032
# P(Baixo) = 6/15 * 1/6 * 2/6 * 2/6 * 1/6 = 0.0012

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base_risco_credito = pd.read_csv('/content/risco_creditoo.csv')

base_risco_credito

X_risco_credito = base_risco_credito.iloc[:,0:4].values
X_risco_credito

Y_risco_credito = base_risco_credito.iloc[:,4].values
Y_risco_credito

from sklearn.preprocessing import LabelEncoder
label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantias = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:,0] = label_encoder_historia.fit_transform(X_risco_credito[:,0]) 
X_risco_credito[:,1] = label_encoder_divida.fit_transform(X_risco_credito[:,1]) 
X_risco_credito[:,2] = label_encoder_garantias.fit_transform(X_risco_credito[:,2]) 
X_risco_credito[:,3] = label_encoder_renda.fit_transform(X_risco_credito[:,3]) 

X_risco_credito
#Resposta
# array([[2, 0, 1, 0],
       [1, 0, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 2],
       [1, 1, 1, 2],
       [1, 1, 0, 2],
       [2, 1, 1, 0],
       [2, 1, 0, 2],
       [0, 1, 1, 2],
       [0, 0, 0, 2],
       [0, 0, 1, 0],
       [0, 0, 1, 1],
       [0, 0, 1, 2],
       [2, 0, 1, 1]], dtype=object)
       
import pickle
with open('risco_credito.pkl','wb') as f:
  pickle.dump([X_risco_credito,Y_risco_credito],f)
  
# Começando Naive Bayes

from sklearn.naive_bayes import GaussianNB 

# historia boa (0), divida alta (0) , garantia nenhuma (1) , renda > 35 (2)
# historia ruim (2), divida alta (0), garantia adequada (0), renda < 15 (0)
previsao = naive_risco_credito.predict([ [0,0,1,2] , [2,0,0,0] ]) 

previsao # prevendo os dois cenários. Resposta : 1 - Baixo , 2 - Moderado

naive_risco_credito.classes_ # ver classes 

naive_risco_credito.class_count_ # Ver quanto Alto , Baixo e Moderado tem na tabela
