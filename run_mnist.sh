

# Weight training
:<<BLOCK
python mnist_neurips.py --mode training \
--lr 0.001 \
--wd 0.0001 \
--optimizer adam \
--epochs 50 \
--results-filename nips_wt_single_20.csv
BLOCK

# Width+Depth Pruning
#:<<BLOCK
python mnist_neurips.py --mode pruning \
--algo ep \
--lr 0.001 \
--optimizer sgd \
--epochs 50 \
--ratio 1 \
--results-filename nips_binary_1.csv

:<<BLOCK
python mnist_neurips.py --mode pruning \
--algo ep \
--lr 0.001 \
--optimizer sgd \
--epochs 50 \
--ratio 2 \
--results-filename nips_binary_2.csv

python mnist_neurips.py --mode pruning \
--algo ep \
--lr 0.001 \
--optimizer sgd \
--epochs 50 \
--ratio 3 \
--results-filename nips_binary_3.csv

python mnist_neurips.py --mode pruning \
--algo ep \
--lr 0.001 \
--optimizer sgd \
--wd 0.0005 \
--epochs 50 \
--ratio 4 \
--results-filename nips_binary_4.csv
BLOCK

:<<BLOCK
python mnist_neurips.py --mode pruning \
--algo ep \
--lr 0.001 \
--optimizer sgd \
--epochs 50 \
--ratio 5 \
--results-filename nips_binary_5.csv
BLOCK


:<<BLOCK
python mnist_neurips.py --mode pruning \
--algo ep \
--lr 0.01 \
--optimizer sgd \
--epochs 50 \
--ratio 10 \
--results-filename nips_binary_10.csv
BLOCK


# (only) Width Pruning
:<<BLOCK
python mnist_neurips.py --mode pruning \
--algo ep \
--lr 0.001 \
--optimizer sgd \
--epochs 50 \
--ratio 1 \
--width-only \
--results-filename nips_binary_width_only_1.csv

python mnist_neurips.py --mode pruning \
--algo ep \
--lr 0.001 \
--optimizer sgd \
--epochs 50 \
--ratio 2 \
--width-only \
--results-filename nips_binary_width_only_2.csv

python mnist_neurips.py --mode pruning \
--algo ep \
--lr 0.001 \
--optimizer sgd \
--epochs 50 \
--ratio 3 \
--width-only \
--results-filename nips_binary_width_only_3.csv


python mnist_neurips.py --mode pruning \
--algo ep \
--lr 0.001 \
--optimizer sgd \
--epochs 50 \
--ratio 4 \
--width-only \
--results-filename nips_binary_width_only_4.csv
BLOCK


:<<BLOCK
python mnist_neurips.py --mode pruning \
--algo ep \
--lr 0.001 \
--optimizer sgd \
--epochs 50 \
--ratio 5 \
--width-only \
--results-filename nips_binary_width_only_5.csv
BLOCK

:<<BLOCK
python mnist_neurips.py --mode pruning \
--algo ep \
--lr 0.01 \
--optimizer sgd \
--epochs 50 \
--ratio 10 \
--width-only \
--results-filename nips_binary_width_only_10.csv
BLOCK

