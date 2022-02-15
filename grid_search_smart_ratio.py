#grid_search_smart_ratio.py
import pandas as pd
import pdb
import csv
import copy



def sum_list(list1, list2):
	zipped_lists = zip(list1, list2)
	return [x + y for (x, y) in zipped_lists]



sparsity_candidate = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]


# 1. load imp sparsity pattern
#imp_sp = pd.read_csv("per_layer_sparsity_resnet20/imp.csv")
#init_sp = imp_sp["1.44"].tolist()

hc_sp = pd.read_csv("per_layer_sparsity_resnet20/hc_iter.csv")
init_sp = hc_sp["1_4"].tolist()

print(init_sp)

# 2. Find possible sparsity pattern and save it (restriction: p_i can be three nearest neighbor of p_i^{IMP})

i = 10
arr_10 = [
					[+i, -i], 
					[0, 0], 
					[-i, +i]
					]
i = 1
arr_1 = [
					[+i, +i, +i, 0, -i, -i, -i], 
					[+i, +i, 0, 0, 0, -i, -i],
					[+i, 0, 0, 0, 0, 0, -i],
					[0, 0, 0, 0, 0, 0, 0]
				]
i = 0.5
arr_0_5 = [
						[+i, +i, +i, -i, -i, -i], 
						[+i, +i, 0, 0, -i, -i],
						[+i, 0, 0, 0, 0, -i],
						[0, 0, 0, 0, 0, 0]
					]
i = 0.1
arr_0_1 = [
						[+i, +i, 0, -i, -i], 
						[+i, 0, 0, 0, -i],
						[0, 0, 0, 0, 0]
					]

result_list = []
for i in range(len(arr_10)):
	for j in range(len(arr_1)):
		for k in range(len(arr_0_5)):
			for l in range(len(arr_0_1)):
				curr_sp = copy.deepcopy(init_sp)

				assert sum(arr_10[i]) == 0
				assert sum(arr_1[j]) == 0
				assert sum(arr_0_5[k]) == 0
				assert sum(arr_0_1[l]) == 0

				#print(curr_sp)

				# update 1st & last layer
				curr_sp[0] += arr_10[i][0] 
				curr_sp[-1] += arr_10[i][1]
				#print(curr_sp)

				# update for 7 layers
				curr_sp[1:8] = sum_list(curr_sp[1:8], arr_1[j])
				#print(curr_sp)

				# update for 6 layers
				curr_sp[8:14] = sum_list(curr_sp[8:14], arr_0_5[k])
				#print(curr_sp)

				# update for 5 layers
				curr_sp[14:19] = sum_list(curr_sp[14:19], arr_0_1[l])
				#print(curr_sp)


				#print(len(curr_sp))
				# save it in the list
				result_list.append(curr_sp)

unit = 24
root = "per_layer_sparsity_resnet20/"
for i in range(6):
	with open(root + "grid_search_{}.csv".format(i), "w") as f:
		write = csv.writer(f)
		write.writerows(result_list[int(unit*i):int(unit*(i+1))])


# with open("per_layer_sparsity_resnet20/grid_search.csv", "w") as f:
# 	write = csv.writer(f)
# 	write.writerows(result_list)


