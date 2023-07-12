import sys
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# SETTING GLOBAL PLOTTING PARAMETERS
sns.set_theme(style="darkgrid", palette="deep", color_codes='true')
FONT_SIZE = 20
LEGEND_FONT_SIZE = 16
TICK_LABEL_SIZE = 14
plt.rc('text', usetex=False)
params = {'legend.fontsize'     : LEGEND_FONT_SIZE,
          'legend.title_fontsize': LEGEND_FONT_SIZE,
		  'axes.labelsize'      : FONT_SIZE,
		  'axes.titlesize'      : FONT_SIZE,
		  'xtick.labelsize'     : TICK_LABEL_SIZE,
		  'ytick.labelsize'     : TICK_LABEL_SIZE,
		  'figure.autolayout'   : True,
          'axes.labelweight'    : 'bold',
          'font.weight'         : 'normal'
         }
plt.rcParams.update(params)

# LOADING DATA
df= pd.read_pickle("quantitative.pkl")
print(df)

# PLOTTING DATA
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x='time delay', y='end distance', data=df, order=None, hue='Threshold', ax=ax)
ax.set_xticklabels(['Constant\nDelay: 0','Constant\nDelay: 1','Constant\nDelay: 2','Constant\nDelay: 3','Random\nDelay: 3(max)'])
ax.legend(ncol=6, frameon=False, title='Threshold',loc='upper center',bbox_to_anchor=(0.5,1.25),
          title_fontproperties={'weight':'bold'}, alignment='right')
ax.set_ylim(-10,170)
ax.set_ylabel('End Distance (m)')
ax.set_xlabel('')

# VIEWING
plt.tight_layout()
plt.savefig('cc_emp.eps', dpi=300)
plt.savefig('cc_emp.pdf') 