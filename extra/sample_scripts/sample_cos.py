import os

# set manually
x_dims = [1] #,2,8
train_shapes = ['cube'] #,'sphere'
signs = ['no']
fixes = [1]   # doesn't matter since signs=["no"]
test_rs = [4.0, 10.0] #original: [4.0, 2.0]
train_rs = [1.0,2.0,4.0]
n_trains = [20000,2000] #oroginal [20000]
n_val = 1000
n_test = 2000
A_rs = [1.0]
b_rs = [0.0]

data = 'cospix' #cos
folder = 'data/non-linear/cospix' #cos
for x_dim in x_dims:
	for train_shape in train_shapes:
		for sign in signs:
			for fix in fixes:
				if sign == 'no' and not fix == 1:
					continue
				if sign == 'z' and not fix == 1:
					continue
				for test_r in test_rs:
					for train_r in train_rs:
						for n_train in n_trains:
							for A_r in A_rs:
								for b_r in b_rs:
									val_r = train_r
									if train_shape == 'cube':
										test_shape = 'cube'
									else:
										test_shape = 'ball'

									os.system("python data_generation.py --folder=%s --data=%s --x_dim=%s --train_shape=%s --test_shape=%s --sign=%s  --fix=%s --test_r=%s --train_r=%s --val_r=%s --n_train=%s --n_val=%s --n_test=%s  --A_r=%s --b_r=%s"%(folder, data, x_dim, train_shape, test_shape, sign, fix, test_r, train_r, val_r, n_train, n_val, n_test, A_r, b_r))
