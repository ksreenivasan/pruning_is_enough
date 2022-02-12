import numpy as np
import pandas as pd


r10 = [10, 
0.32175925925925924,
0.0920138888888889,
0.07682291666666667,
0.08159722222222222,
0.05555555555555555,
0.053385416666666664,
0.05078125,
0.042534722222222224,
0.0361328125,
0.03428819444444445,
0.027560763888888888,
0.021267361111111112,
0.014105902777777778,
0.01416015625,
0.010552300347222222,
0.007378472222222222,
0.005343967013888889,
0.0028483072916666665,
0.001681857638888889,
0.2640625
]

r30 = [30,
0.33101851851851855,
0.09722222222222222,
0.0876736111111111,
0.06770833333333333,
0.07074652777777778,
0.0642361111111111,
0.046440972222222224,
0.04296875,
0.038736979166666664,
0.027886284722222224,
0.026258680555555556,
0.023980034722222224,
0.016493055555555556,
0.01416015625,
0.010850694444444444,
0.007134331597222222,
0.005533854166666667,
0.002685546875,
0.0023057725694444445,
0.265625
]

r70 = [70,
0.29398148148148145,
0.09418402777777778,
0.09505208333333333,
0.0720486111111111,
0.0681423611111111,
0.055989583333333336,
0.04600694444444445,
0.04513888888888889,
0.035698784722222224,
0.03342013888888889,
0.027018229166666668,
0.0205078125,
0.015516493055555556,
0.013292100694444444,
0.010118272569444444,
0.007080078125,
0.006076388888888889,
0.002875434027777778,
0.002387152777777778,
0.2375
]

r160 = [160,
0.35648148148148145,
0.08203125,
0.08637152777777778,
0.0685763888888889,
0.06684027777777778,
0.059895833333333336,
0.047309027777777776,
0.044921875,
0.03634982638888889,
0.03309461805555555,
0.027126736111111112,
0.023328993055555556,
0.017469618055555556,
0.012803819444444444,
0.011013454861111112,
0.00732421875,
0.006537543402777778,
0.003146701388888889,
0.003200954861111111,
0.203125
]



np.savetxt('smart_ratio_v3_lr1e-8_manual.csv', [p for p in zip(r10, r30, r70, r160)], delimiter=',', fmt='%s')