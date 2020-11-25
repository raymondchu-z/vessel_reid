import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='ap_curve')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
opt = parser.parse_args()



ap_path = './model/%s/ap.csv'%opt.name
ap_df = pd.read_csv(ap_path)

####
plt.figure()
data = pd.Series(ap_df["ap"])
ax = data.hist(bins = 50) 
ax.set_xlabel('ap')
ax = data.plot(kind="kde",secondary_y = True, xlim=(-0.1,1.1))
# plt.legend()
ax.figure.savefig('./model/%s/ap_curve.png'%opt.name)