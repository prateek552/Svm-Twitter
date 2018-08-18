
import pandas
df=pandas.read_excel("tweet.xlsx")
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['tweet'], df['label'],test_size=0.2)
df2 = pandas.DataFrame({'tweet': train_x, 'label': train_y})
df2.fillna(1)
df2.to_csv("train.csv",index=False)
df4=pandas.read_csv("train.csv")
print (df4)
df3 = pandas.DataFrame({'tweet': valid_x, 'label': valid_y})
df3.fillna(1)
df3.to_csv("test.csv",index=False)
df4=pandas.read_csv("test.csv")

