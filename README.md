This repository is an offshoot of TensorFlow's CIFAR10 tutorial (http://tensorflow.org/tutorials/deep_cnn/). It has been repurposed for experimenting with Knowledge Distillation (https://arxiv.org/pdf/1503.02531.pdf). It was done primarily in TensorFlow v0.7 so a proper update might be necessary to make compatible with more recent releases.

Modifications have been made to cifar10.py, cifar10_input.py, and cifar10_eval.py, while cifar10_student.py is heavily built on the train module. In short, a "teacher" net that classifies images is trained, and replica or smaller "student" nets are trained on the logit outputs of the teacher net and sometimes other student nets.

CIFAR-10 is a common benchmark dataset in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html
