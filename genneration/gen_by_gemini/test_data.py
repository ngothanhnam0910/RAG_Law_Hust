import pandas as pd

path_csv = "/point/namnt/DATN/genneration/data/qa_output.csv"

df = pd.read_csv(path_csv)
print(df.head(10))
