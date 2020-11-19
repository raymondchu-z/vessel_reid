""" 
根据ap.csv文件，计算IMO的平均AP，保存在IMO_ap.csv里
"""
import pandas as pd
model_path = "/home/zlm/research/Person_reID_baseline_pytorch/model/ft_192_lr08"
ap_df = pd.read_csv(model_path +"/ap.csv")
IMO_group = ap_df['ap'].groupby(ap_df['label'])
# print(IMO_group.mean().sort_values())
df = pd.DataFrame(IMO_group.mean().sort_values())
df.to_csv(model_path + '/IMO_ap.csv')

  