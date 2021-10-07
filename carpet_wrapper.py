# ====== Module Import ====== #

import sys; sys.path.insert(0, '.')
import torch
import numpy as np

class model_wrapper:
    def __init__(self, model, window_size, device='cuda:0'):
        self.model = model
        self.device = device
        self.input_buffer = []
        self.window_size = window_size
        self.foot_gesture = ['default', 'stand', 'tap'] # label name

    def predict(self, frame):

        self.input_buffer += frame

        if len(self.input_buffer) > self.window_size and len(self.input_buffer) % 15 == 0 :
            print(len(self.input_buffer))
            input_frame = torch.Tensor(np.array(self.input_buffer[-self.window_size:]).astype(np.uint8)).cuda()
            input_frame = input_frame.unsqueeze(0)
            input_frame = input_frame.reshape(1, self.window_size, 1, 32, 32).float()

            self.model.cuda()
            motion = self.model(input_frame)
            motion = list(motion.cpu().detach().numpy()[0])
            length = len(motion)
            log = ""
            for i in range(0, length):
                m = motion.index(max(motion))
                log = log + str(self.foot_gesture[m]) + "(" + str(motion[m]) + ")     "
                motion[m] = -10000

            print(log)

