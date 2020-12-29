import csv
import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from numpy import arange

no_class_imb_cor_filename = 'results_mul/hier_fuse~elmo~rnn~11~~glove~rnn~21~~fasttext~rnn~31~~ling~rnn~41~~bert_pre~1~use~1~infersent~1~0_1_2_3_4_5_6_7_8_9_10_11_12_13~di~False~100~2_3_4~lstm~100~100~1~False~True.txt'
class_imb_cor_filename = 'results_mul/hier_fuse~elmo~rnn~11~~glove~rnn~21~~fasttext~rnn~31~~ling~rnn~41~~bert_pre~1~use~1~infersent~1~0_1_2_3_4_5_6_7_8_9_10_11_12_13~di~True~100~2_3_4~lstm~200~400~1~False~True.txt'
class_imb_with_without_filename = 'results_mul/class_imb_analysis_with_without_filename.txt'
output_filename = 'results_mul/class_imb_bar_chart_with_without_filename.pdf'

paper_class_mapping = {'Role stereotyping':1,
'Attribute stereotyping':2,
'Body shaming':3,
'Hyper-sexualization (excluding body shaming)':4,
'Internalized sexism':5,
'Hostile work environment':6,
'Denial or trivialization of sexist misconduct':7,
'Threats':8,
'Sexual assault':9,
'Sexual harassment (excluding assault)':10,
'Moral policing and victim blaming':11,
'Slut shaming':12,
'Motherhood and menstruation related discrimination':13,
'Other':14
}

no_cl_imb_cor_csv = csv.DictReader(open(no_class_imb_cor_filename), delimiter = '\t')
cl_imb_cor_csv = csv.DictReader(open(class_imb_cor_filename), delimiter = '\t')
NUM_CLASSES = 0
l = []
for no_cl_row, cl_row in zip(no_cl_imb_cor_csv, cl_imb_cor_csv):
	assert cl_row['lab id'] == no_cl_row['lab id']
	assert cl_row['label'] == no_cl_row['label']
	row = {'lab id paper': paper_class_mapping[cl_row['label']], 'label': cl_row['label'], 'train cov': cl_row['train cov']}
	row['F score without class imbalance correction'] = no_cl_row['F score']
	row['F score with class imbalance correction'] = cl_row['F score']
	l.append(row)
	NUM_CLASSES += 1
# print(l)
l = sorted(l, key=lambda k: float(k['train cov']))
with open(class_imb_with_without_filename, 'w') as f_fin:
	w_fin = csv.DictWriter(f_fin, fieldnames = ['lab id paper', 'label', 'train cov', 'F score without class imbalance correction', 'F score with class imbalance correction'], delimiter = ',')
	w_fin.writeheader()
	for row in l:
		w_fin.writerow(row)

textDF=pandas.read_csv(class_imb_with_without_filename)
ours = 'F score with class imbalance correction'
competitor = 'F score without class imbalance correction'
fig, ax = plt.subplots(figsize=(8,5))
print(type(textDF))
print(type(textDF[competitor]))
textDF[competitor].plot.bar(width=0.2, ylim=[0, 1.0], position=1, color="blue", ax=ax, alpha=1)
textDF[ours].plot.bar(width=0.2, position=0, ylim=[0, 1.0],color="violet", ax=ax, alpha=1)
ax.set_facecolor("white")
# ax.set_xticks(range(0,NUM_CLASSES))
ax.set_xticklabels(list(textDF['lab id paper']), rotation=0, fontsize=7)
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
ax.set_yticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],rotation=0, fontsize=7)
ax.set_xlabel("Label IDs",fontsize=7)	

for i in range(NUM_CLASSES):
    plt.text(x = i-0.2 , y = max(textDF[competitor][i],textDF[ours][i])+.0125, s = ("%.1f" % textDF["train cov"][i]), size = 7,rotation=0, bbox=dict(facecolor='lightgray', edgecolor='none',alpha=1, pad=1))

tr_c = mpatches.Patch(color='lightgray', label='Label coverage %')
best_base = mpatches.Patch(color='blue', label=competitor)
best_proposed = mpatches.Patch(color='violet', label=ours)
ax.legend(handles=[tr_c,best_base,best_proposed],loc="upper right", prop={'size': 7},bbox_to_anchor=(1.0035,1.007))
plt.savefig(output_filename,bbox_extra_artists=(ax,), bbox_inches='tight')

