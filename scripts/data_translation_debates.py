


import pandas as pd
pd.options.mode.chained_assignment = None
from googletrans import Translator

translator = Translator()

# call the selected comments into a dataframe
df = pd.read_excel('/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/madrid_data/debates.xlsx')

# translate the "title" column
df['title_en'] = ''
for i in range(len(df['title'])):
    t = df['title'][i]
    if t:
        try:
            trans = translator.translate(t, src='es', dest='en')
            if trans:
                df['title_en'][i] = trans.text
        except Exception as e:
            print(str(e))
            continue

# translate the "description" column
df['description_en'] = ''
for i in range(len(df['description'])):
    t = df['description'][i]
    if t:
        try:
            trans = translator.translate(t, src='es', dest='en')
            if trans:
                df['description_en'][i] = trans.text
        except Exception as e:
            print(str(e))
            continue

# save the modified DataFrame as a new Excel file
df.to_excel('/Users/cem_ataman/PycharmProjects/MadridDecidim-Data-Analysis/data/madrid_data/debates_en.xlsx', index=False)
