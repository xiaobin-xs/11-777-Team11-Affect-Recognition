import numpy as np
base_dir = 'drive/MyDrive/DEAP/'

face_features_url = base_dir + '/face_embeddings.txt'
video_features_url = base_dir + '/new_video_embeddings.txt'
labels_url = base_dir + '/labels_16.txt'

# import numpy as np
face_dataset = np.loadtxt(face_features_url)
video_dataset = np.loadtxt(video_features_url)

features_df = np.concatenate( (face_dataset,video_dataset), axis = 1 )
labels_df = np.loadtxt(labels_url)

from sklearn.model_selection import train_test_split

X_train_agg, X_test_agg, y_train_agg, y_test_agg = train_test_split(features_df, labels_df, test_size=0.3, random_state=42)
X_train_face, X_test_face, y_train_face, y_test_face = train_test_split(face_dataset, labels_df, test_size=0.3, random_state=42)
X_train_vid, X_test_vid, y_train_vid, y_test_vid = train_test_split(video_dataset, labels_df, test_size=0.3, random_state=42)

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

def runModel(train_x, train_y, test_x, test_y, modal_title, epochs = 1500, input_dim = 10):
  model = nn.Sequential(nn.Linear(input_dim, 40),nn.ReLU(), nn.Linear(40, input_dim), nn.ReLU(), 
                        nn.Linear(input_dim, 4))

  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  losses, accs = [], []

  for epoch in tqdm(range(epochs)):
      optimizer.zero_grad()
      output = model(torch.from_numpy(train_x).type(torch.FloatTensor)).type(torch.FloatTensor)
      #print(output)
      loss = criterion(output, torch.from_numpy(train_y))
      loss.backward()
      optimizer.step()
      
      res = model(torch.from_numpy(test_x).type(torch.FloatTensor))
      res = torch.FloatTensor([[1 if x >= 0 else 0 for x in item] for item in res])
      acc =  1 - sum(abs(res.flatten() - test_y.flatten()))/len(res.flatten())
      accs.append(acc)
      losses.append(loss.item())

  plt.title(modal_title + ' Accuracy')
  plt.plot(range(epochs), accs )
  plt.show()

  plt.title(modal_title + ' Loss')
  plt.plot(range(epochs), losses )
  plt.show()

  print('F1: ', f1_score(test_y, res, average='weighted'), ' acc: ', accs[-1])

  runModel(X_train_face, y_train_face, X_test_face, y_test_face, 'Facial Embedding' )

  runModel(X_train_vid, y_train_vid, X_test_vid, y_test_vid, 'Video Embedding' )

  runModel(X_train_agg, y_train_agg, X_test_agg, y_test_agg, 'Aggregate Embedding',  input_dim = 20 )