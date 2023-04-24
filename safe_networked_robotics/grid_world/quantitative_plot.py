import sys
sys.path.remove('/usr/lib/python3/dist-packages')
import seaborn as sns
import matplotlib.pyplot as plt

# Create a sample dataframe
df = sns.load_dataset('titanic')

# Create the figure and axes
fig, ax = plt.subplots()

# Plot the grouped bar plot
sns.barplot(x='sex', y='survived', hue='class', data=df, dodge=False, ax=ax)

# Stack the bars
for i, bar in enumerate(ax.containers):
    for j, patch in enumerate(bar.get_children()):
        if j == 0:
            patch.set_width(.5)
            patch.set_x(patch.get_x() - .25)
        else:
            patch.set_width(.5)
            patch.set_x(patch.get_x() + .25)

plt.savefig('sample.png')