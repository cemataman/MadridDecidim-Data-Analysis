import pandas as pd
pd.options.mode.chained_assignment = None
from googletrans import Translator

translator = Translator()

# call the selected comments into a dataframe
df = pd.read_excel('/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/intermediate_data/selected_debate_comments.xlsx')
df_es = df.iloc[:,[3]] # comments section

# in order to send only one row at a time to google translator to remain within the character limit of google translator
### IT TAKES A LONG TIME IF THE DATASET IS BIG ###
for i in range(len(df_es['body'])):
    t = df_es['body'][i]
    if t:
        try:
            trans = translator.translate(t, src='es', dest='en')
            if trans:
                df_es['body'][i] = trans.text
        except Exception as e:
            print(str(e))
            continue
df["body"] = df_es

# save the translated data as an excel file
df.to_excel("/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/intermediate_data/selected_debate_comments_en.xlsx", index=False)
