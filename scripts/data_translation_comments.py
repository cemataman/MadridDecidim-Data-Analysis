import pandas as pd
pd.options.mode.chained_assignment = None
from googletrans import Translator

translator = Translator()

# call the selected comments into a dataframe
df = pd.read_excel('/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/madrid_data/comments.xlsx')
df_es = df.iloc[:,[3]] # comments section

# Create an empty column "body_en" to store the translated comments
df["body_en"] = ""

# Translate each comment in the "body" column and store it in the "body_en" column
for i, t in enumerate(df_es['body']):
    if t:
        try:
            trans = translator.translate(t, src='es', dest='en')
            if trans:
                df["body_en"][i] = trans.text
        except Exception as e:
            print(str(e))
            continue

# Save the translated DataFrame as a new Excel file
df.to_excel('/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/madrid_data/comments_en.xlsx', index=False)
