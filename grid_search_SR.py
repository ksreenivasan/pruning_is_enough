'''
grid search for SR
'''

import os
import glob
import pandas as pd
import pdb

# use glob to get all the csv files in the folder

best_idx = 0
best_acc = 0
best_sp = 0
for idx in range(1, 145):

	path = 'results/SR_grid_sp_1_44_{}'.format(idx)
	csv_files = glob.glob(os.path.join(path, "*/**.csv"))
	#print(idx)

	# loop over the list of csv files
	for f in csv_files:

		df = pd.read_csv(f)
		#print('Location:', f)
		#print('File Name:', f.split("\\")[-1])
		curr_acc = df['test_acc'].tolist()[-1]
		curr_sp = df['model_sparsity'].tolist()[-1]
		print('idx: {}, acc: {}, sparsity: {:.2f}'.format(idx, curr_acc, curr_sp))
		#pdb.set_trace()
		if curr_acc > best_acc:
			best_idx = idx
			best_acc = curr_acc
			best_sp = curr_sp

print('best_idx: {}, best_acc: {}, best_sparsity: {:.2f}'.format(best_idx, best_acc, best_sp))


# save the optimal grid search result (sparsity pattern)
#best_idx = 113 # for DEBUG
group = int(best_idx/24)
line_idx = best_idx % 24

PATH = 'per_layer_sparsity_resnet20/grid_search_saved_{}.csv'.format(group)
df = pd.read_csv(PATH)
if line_idx == 0:
    best_pattern = df.columns.tolist()
else:
    best_pattern = df.values.tolist()[line_idx-1]

print(best_pattern)

