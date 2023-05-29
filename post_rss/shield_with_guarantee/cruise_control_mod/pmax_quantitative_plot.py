import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# SETTING GLOBAL PLOTTING PARAMETERS
sns.set_theme(style="darkgrid", palette="deep", color_codes='true')
FONT_SIZE = 18
LEGEND_FONT_SIZE = 14
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
df= pd.read_pickle("pmax.pkl")
print(df)

# PLOTTING DATA
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x='Threshold', y='maximum safety prob', data=df, order=None, hue='time delay', ax=ax)
#ax.set_xticklabels(['Constant\nDelay: 0','Constant\nDelay: 1','Constant\nDelay: 2','Constant\nDelay: 3','Random\nDelay: 3(max)'])
ax.set_xticklabels(['delta: 0.5','delta: 0.8','delta: 0.9','delta: 0.95'])
ax.legend(ncol=6, frameon=False, title='Time delay',loc='upper center',bbox_to_anchor=(0.5,1.25), alignment='left',
          title_fontproperties={'weight':'bold'})
ax.set_ylim(-0.1,1.0)
ax.set_ylabel('Min reach probability')
ax.set_xlabel('')

#x = ['delta: 0.5','delta: 0.8','delta: 0.9','delta: 0.95']
#y = [1-0.5, 1-0.8, 1-0.9, 1-0.95]
#line_df = pd.DataFrame([])
#line_df['delta'] = x 
#line_df['prob'] = y 
#sns.lineplot(x = "delta", y = "prob", data=line_df)


# VIEWING
plt.tight_layout()
plt.savefig('cc_pmax_emp.pdf')