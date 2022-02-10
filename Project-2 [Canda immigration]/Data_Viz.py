import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pywaffle import Waffle
from wordcloud import WordCloud, STOPWORDS
import wikipedia

df_c_imm = pd.read_excel("C:\\Users\\Asus\\Downloads\\Cog =2\\Canada.xlsx", sheet_name='Canada by Citizenship',
                         skiprows=range(20),
                         skipfooter=2)
print(df_c_imm)
# print(f"{df_c_imm.index} \n {df_c_imm.columns} \n Info \n {df_c_imm.info()}")

check = df_c_imm.isnull().sum()
print(f"Missing Values \n {check}")
# No Missing Values

stat = df_c_imm.describe(include='all')
print(f"Descriptive Analysis \n {stat}")

df_c_imm.columns = list(map(str, df_c_imm.columns))

df_c_imm.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)
print(df_c_imm.head())

df_c_imm.rename(
    columns={'OdName': 'Country', 'AreaName': 'Continent', 'RegName': 'Region', 'DevName': 'Type of Region'},
    inplace=True)
print(df_c_imm.head())
df_map = df_c_imm

print(df_c_imm['Country'].value_counts())

df_c_imm['Total'] = df_c_imm.loc[:, '1980':'2013'].sum(axis=1)
print(df_c_imm.head())

df_c_imm.set_index('Country', inplace=True)
print(df_c_imm.head())

# data on Asian countries and region = southern asia
confilt = ((df_c_imm.Continent == 'Asia') & (df_c_imm.Region == 'Southern Asia'))
print(df_c_imm[confilt])

ind = df_c_imm.index.tolist()
col = df_c_imm.columns.tolist()
print(f"Index  = {ind} \n columns = {col}")

years = list(map(str, range(1980, 2014, 1)))
# --------------------------------------------------------------------------------------------------------------------------------
# Viz using Line plots

# In 2010, Haiti suffered a catastrophic magnitude 7.0 earthquake. The quake caused widespread devastation and loss of life
# and about three million people were affected by this natural disaster. As part of Canada's humanitarian effort,
# the Government of Canada stepped up its effort in accepting refugees from Haiti. We can quickly visualize this effort
# using a Line plot:
# Question - Plot a line graph of immigration from Haiti
Window1 = plt.figure(1)
Haiti_DF = df_c_imm.loc['Haiti', years]
print(Haiti_DF)
Haiti_DF.index = Haiti_DF.index.map(int)
Haiti_DF.plot(kind='line')
plt.xlabel('Years')
plt.ylabel('Immigrants')
plt.title('immigration from Haiti to Canada')
plt.text(2002, 6002, '2010 Earthquake')

# India and china
Window2 = plt.figure(2)
India_DF = df_c_imm.loc['India', years]
print(India_DF)
India_DF.index = India_DF.index.map(int)
India_DF.plot(kind='line')
plt.xlabel('Years')
plt.ylabel('Immigrants')
plt.title('immigration from India to Canada')

Window3 = plt.figure(3)
China_DF = df_c_imm.loc['China', years]
print(China_DF)
China_DF.index = China_DF.index.map(int)
China_DF.plot(kind='line')
plt.xlabel('Years')
plt.ylabel('Immigrants')
plt.title('immigration from China to Canada')

Window4 = plt.figure(4)
Ind_Chin_DF = df_c_imm.loc[['India', 'China'], years]
print(Ind_Chin_DF)
Ind_Chin_DF.T.plot(kind='line')
plt.xlabel('Years')
plt.ylabel('Immigrants')
plt.title('immigration from India and China to Canada')

df_c_imm.sort_values(ascending=False, by='Total', inplace=True)
print(df_c_imm.head())

Window5 = plt.figure(5)
df_top5 = df_c_imm.loc['India':'Pakistan', years]
print(df_top5)
df_top5.T.plot(kind='area', stacked=False, alpha=0.25)  # aplha - transparency of unstacked area plot
plt.xlabel('Years')  # Scripting layer
plt.ylabel('Immigrants')
plt.title('immigration from Top 5 Countries Contributing to Canada')
# ax = df_top5.T.plot() - artist layer
# ax.xlabel and all that


Window7 = plt.figure(7)
count, bin_mark = np.histogram(df_c_imm['2013'])
df_c_imm['2013'].plot(kind='hist')
plt.xticks(bin_mark)
plt.xlabel('Number of Immigrants')
plt.ylabel('Number of Countries')
plt.title("Immigration in 2013")
plt.show(block=False)
plt.close('all')

Window9 = plt.figure(9)
df_3 = df_c_imm.loc[['Denmark', 'Norway', 'Sweden'], years]
count1, bin_wid = np.histogram(df_3.T)
df_3.T.plot(kind='hist', alpha=0.35, color=['Red', 'Green', 'Blue'], stacked=True)
plt.xticks(bin_wid)
plt.xlabel('Number of Immigrants')
plt.ylabel('Number of Years')
plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')

# immigration from iceland to canada
Window10 = plt.figure(10)
df_ice = df_c_imm.loc['Iceland', years]
df_ice.plot(kind='bar')
plt.title("Immigration from Iceland to China ")
plt.xlabel("Years")
plt.ylabel("Immigrants")
# Annotate arrow
plt.annotate('',  # s: str. will leave it blank for no text
             xy=(32, 70),  # place head of the arrow at point (year 2012 , pop 70)
             xytext=(28, 20),  # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',  # will use the coordinate system of the object being annotated
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
             )

# Annotate Text
plt.annotate('2008 - 2011 Financial Crisis',  # text to display
             xy=(28, 30),  # start the text at at point (year 2008 , pop 30)
             rotation=72.5,  # based on trial and error to match the arrow
             va='bottom',  # want the text to be vertically 'bottom' aligned
             ha='left',  # want the text to be horizontally 'left' algned.
             )

# pie -
# autopct - is a string or function used to label the wedges with their numeric value. The label will be placed inside the wedge.
# If it is a format string, the label will be fmt%pct.
# startangle - rotates the start of the pie chart by angle degrees counterclockwise from the x-axis.
# shadow - Draws a shadow beneath the pie (to give a 3D feel).

# Remove the text labels on the pie chart by passing in legend and add it as a seperate legend using plt.legend().
# Push out the percentages to sit just outside the pie chart by passing in pctdistance parameter.
# Pass in a custom set of colors for continents by passing in colors parameter.
# Explode the pie chart to emphasize the lowest three continents (Africa, North America, and Latin America and Caribbean)
# by passing in explode parameter.
df_continent = df_c_imm.groupby('Continent').sum()
colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0, 0, 0.1, 0.1]  # ratio for each continent with which to offset each wedge.

df_continent['Total'].plot(kind='pie',
                           autopct='%1.1f%%',
                           startangle=90,
                           shadow=True,
                           labels=None,  # turn off labels on pie chart
                           pctdistance=1.12,
                           # the ratio between the center of each pie slice and the start of the text generated by autopct
                           colors=colors_list,  # add custom colors
                           explode=explode_list  # 'explode' lowest 3 continents
                           )

# scale the title up by 12% to match pctdistance
plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.12)

plt.axis('equal')

plt.title("Immigration to canada on the basis of Continents", y=1.12)

# add legend
plt.legend(labels=df_continent.index, loc='upper left')

# Boxplot
Ind_Chin_DF.T.plot(kind='box')
print(Ind_Chin_DF.T.describe())
Ind_Chin_DF.T.plot(kind='box', vert=False)

plt.show(block=False)
plt.close('all')

# Scatter Plot
year_tot = pd.DataFrame(df_c_imm[years].sum(axis=0))
year_tot.reset_index(inplace=True)
year_tot.columns = ['Year', 'Total_Count']
year_tot['Year'] = year_tot['Year'].astype(int)
year_tot.plot(kind='scatter', x='Year', y='Total_Count')
plt.title('Total Immigration to Canada from 1980 - 2013')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')

x = year_tot['Year']  # year on x-axis
y = year_tot['Total_Count']  # total on y-axis
fit = np.polyfit(x, y, deg=1)

print(f' best fit line - {fit}')  # ax+b

plt.plot(x, fit[0] * x + fit[1], color='red')  # recall that x is the Years
plt.annotate('y={0:.1f} x + {1:.1f}'.format(fit[0], fit[1]), xy=(2000, 150000))

# Bubble Plot
df_bra_arg = df_c_imm.loc[['Brazil', 'Argentina'], years]
df_bra_arg = df_bra_arg.T
df_bra_arg.reset_index(inplace=True)
df_bra_arg.columns = ['Year', 'Brazil', 'Argentina']
print(df_bra_arg)
df_bra_arg['Year'] = df_bra_arg['Year'].astype(int)

# for 3rd argument for bubble plot -  also known as scale or weight

# normalize Brazil data
norm_brazil = (df_bra_arg['Brazil'] - df_bra_arg['Brazil'].min()) / (
        df_bra_arg['Brazil'].max() - df_bra_arg['Brazil'].min())

# normalize Argentina data
norm_argentina = (df_bra_arg['Argentina'] - df_bra_arg['Argentina'].min()) / (
        df_bra_arg['Argentina'].max() - df_bra_arg['Argentina'].min())

# Brazil
ax0 = df_bra_arg.plot(kind='scatter',
                      x='Year',
                      y='Brazil',
                      alpha=0.5,  # transparency
                      color='green',
                      s=norm_brazil * 2000 + 10,  # pass in weights  #as normalized size would be small
                      xlim=(1975, 2015)
                      )

# Argentina
ax1 = df_bra_arg.plot(kind='scatter',
                      x='Year',
                      y='Argentina',
                      alpha=0.5,
                      color="blue",
                      s=norm_argentina * 2000 + 10,
                      ax=ax0
                      )

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from Brazil and Argentina from 1980 to 2013')
ax0.legend(['Brazil', 'Argentina'], loc='upper left')
plt.show(block=False)
plt.close('all')
# -----------------------------ADVANCE--------------------------------------------------------------------------------------------------
# waffle chart - data represented in the form of collection of boxes of certain length and breadth, depending on the data
# display progress toward goals. It is commonly an effective option when you are trying to add interesting visualization
# features to a visual that consists mainly of cells, such as an Excel dashboard.
##word cloud - More frequent words appear larger
# regplot - scatter plot with fitted line

ind = ((df_c_imm.index == 'Denmark') | (df_c_imm.index == 'Norway') | (df_c_imm.index == 'Sweden'))
df_dns = df_c_imm[ind]
print(df_dns)

# Step 1. The first step into creating a waffle chart is determining the proportion of each category with respect to the total.
total = df_dns['Total'].sum()
category_grp_proportion = df_dns['Total'] / total
cat = pd.DataFrame({'Proportion': category_grp_proportion})
print(cat)

# Step 2. The second step is defining the overall size of the waffle chart.
width = 40  # width of chart
height = 10  # height of chart
total_num_tiles = width * height  # total number of tiles
print(f'Total number of tiles is {total_num_tiles}.')

# Step 3. The third step is using the proportion of each category to determine it respective number of tiles
tiles_per_cat = (category_grp_proportion * total_num_tiles).round().astype(int)
til_cat = pd.DataFrame({'Tile per Category': tiles_per_cat})
print(til_cat)

lis1 = cat.index
lis2 = cat['Proportion']
label = [f'{key} - Proportion {(value * 100)} % ' for key, value in zip(lis1, lis2)]
fig = plt.figure(FigureClass=Waffle, rows=height, columns=width, values=df_dns['Total'],
                 legend={'labels': label, 'loc': (1, 1.2)},
                 colors=['red', 'blue', 'yellow'])
plt.show()

# Word Clouds
with open('C:\\Users\\Asus\\Downloads\\WorkData\\Speech.txt', 'r', encoding='utf-8') as obj:
    text = obj.read()

stopwords = STOPWORDS

wc = WordCloud(background_color='white', stopwords=stopwords, height=500, width=600)
wc.generate(text)

wc.to_file('1st Word Cloud.png')


def get_wiki(query):
    title = wikipedia.search(query)
    page = wikipedia.page(title)
    return page.content


txt = get_wiki('Adolf Hitler')
print(txt)

stopwords = STOPWORDS

wc_1 = WordCloud(background_color='white', stopwords=stopwords, height=600, width=600)
wc_1.generate(txt)

wc_1.to_file('Hitler word cloud.png')