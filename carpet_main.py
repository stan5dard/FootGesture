
# ====== Module Import ====== #
import numpy as np
import torch
import argparse
from copy import deepcopy
import carpet_model
import carpet_dataset
from carpet_datacollect import MultiSensors
import carpet_wrapper
from torchsummary import summary
import torchvision

# ====== Random Seed Initialization ====== #
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)
parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.exp_name = "exp1_lr"
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Regularization ======= #
args.l2 = 0.00001
args.dropout = 0.5
args.use_bn = True

# ====== Optimizer & Training ====== #
args.model = 'LSTM_CNN'
args.class_num = 3
args.path = './dataset/'

# ====== Experiment Variable ====== #
name_var1 = 'optim'
name_var2 = 'batch_size'
name_var3 = 'epoch'
name_var4 = 'lr'
name_var5 = 'window_size'
list_var1 = ['SGD']
list_var2 = [64]
list_var3 = [20]
list_var4 = [0.001]
list_var5 = [40]

md = int(input("Enter the number of mode(1: model training, 2: model loading): "))

if md == 1:
    for var1 in list_var1:
        for var2 in list_var2:
            for var3 in list_var3:
                for var4 in list_var4:
                    for var5 in list_var5:
                        setattr(args, name_var1, var1)
                        setattr(args, name_var2, var2)
                        setattr(args, name_var3, var3)
                        setattr(args, name_var4, var4)
                        setattr(args, name_var5, var5)
                        print('args :', args)

                        # trainset, valset, testset 8 : 1 :1
                        trainset = carpet_dataset.NumDataset(args)
                        train_num = round(len(trainset) * 0.8)
                        print('total num :', len(trainset))
                        print('train_num :', train_num)
                        train_num = train_num // args.batch_size
                        train_num = train_num * args.batch_size
                        val_num = round(len(trainset) * 0.1)
                        print('val_num :', val_num)
                        test_num = len(trainset) - train_num - val_num
                        print('test_num :', test_num)
                        trainset, valset, testset = torch.utils.data.random_split(trainset, [train_num, val_num, test_num])

                        partition = {'train': trainset, 'val': valset, 'test': testset}
                        setting, result = carpet_model.experiment(partition, deepcopy(args))

                        print('Settings:', setting)
                        print('train_accuracy :', result['train_acc'])
                        print('val_accuracy :', result['val_acc'])

elif md == 2:
    window_size = 20
    saved_path = "./trained_model/OpSGDBat64Epo20Lr0001Win40Model2_1007.pt"
    model = carpet_model.LSTM_CNN(args)
    model.to(args.device)
    model.load_state_dict(torch.load(saved_path))
    main_model = carpet_wrapper.model_wrapper(model.eval(), window_size=window_size)

    sensor = MultiSensors()
    print("initializing sensors...")
    sensor.init_sensors()

    # Predicted Gesture Inference
    i = 0
    steady = []
    while True:
        total_images = sensor.get_all()
        foot_gesture = main_model.predict(total_images)
        i += 1

    sensor.close()
else:
    raise ValueError('In-valid mode choice')
