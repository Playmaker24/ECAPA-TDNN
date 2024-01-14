'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import os
import argparse, glob, os, torch, warnings, time
from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel
from sklearn.model_selection import KFold

##tensorboard setup 
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=4,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--path', type=str,   default="dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
#parser.add_argument('--train_list', type=str,   default="/data08/VoxCeleb2/train_list.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
#parser.add_argument('--train_path', type=str,   default="/data08/VoxCeleb2/train/wav",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
#parser.add_argument('--train_list', type=str,   default="dataset/Speech_data_forSSC_copy/Speech_data_forSSC/Speech_data/Emotional_speech/Amazon+Google_voices/Emotional_speech_female_eval_list.txt")
#parser.add_argument('--train_path', type=str,   default="dataset/Speech_data_forSSC_copy/Speech_data_forSSC/Speech_data/Emotional_speech/Amazon+Google_voices/train/wav/female")
#parser.add_argument('--eval_list',  type=str,   default="dataset/vox/veri_test2.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
#parser.add_argument('--eval_path',  type=str,   default="dataset/vox/vox1_test_wav/wav",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--musan_path', type=str,   default="dataset/musan_split",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path',   type=str,   default="dataset/rirs_noises/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
parser.add_argument('--speech_type', type = str, default="Neutral_speech", help='The speech type to train [Neutral_speech vs Emotional_speech]')
parser.add_argument('--gender', type = str, default="", help='The gender to train [male vs female]')

parser.add_argument('--save_path',  type=str,   default="exps/exp1",                                     help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')

## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
#parser.add_argument('--n_class', type=int,   default=8,   help='Number of speakers')
parser.add_argument('--n_class', type=int,   default=4,   help='Number of unique emotions')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)


## Modified new data loader for data_2.0
#train_list_path = os.path.join(args.path, args.speech_type, "Amazon+Google_voices", args.speech_type + "_" + "train_list.txt")
train_list_path = os.path.join(args.path, args.speech_type, "Amazon+Google_voices", args.gender + "_train_list_one_hot.txt")
train_path = os.path.join(args.path, args.speech_type, "Amazon+Google_voices/train/wav", args.gender)
musan_path = args.musan_path
rir_path = args.rir_path

trainloader = train_loader(train_list=train_list_path, train_path=train_path, musan_path=musan_path, rir_path=rir_path, num_frames=args.num_frames)
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)


"""
## Modified new data loader
train_list_path = os.path.join(args.path, args.speech_type, "Amazon+Google_voices", args.speech_type + "_" + args.gender + "_train_list.txt")
train_path = os.path.join(args.path, args.speech_type, "Amazon+Google_voices/train/wav", args.gender)
musan_path = args.musan_path
rir_path = args.rir_path

trainloader = train_loader(train_list=train_list_path, train_path=train_path, musan_path=musan_path, rir_path=rir_path, num_frames=args.num_frames)
#print("trainLoader content 1:", len(trainloader))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
#print("trainLoader content 2:", len(trainLoader))
"""

## Define the data loader
#trainloader = train_loader(**vars(args))
#trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

##Define SummaryWriter from Tensorboard to visualize the acc and loss of the model
writer = SummaryWriter()

## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	#EER, minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
	#EER, minDCF = s.eval_network(eval_list = os.path.join(args.path, args.speech_type, "Amazon+Google_voices", args.speech_type + "_" + args.gender + "_eval_list.txt"), eval_path = os.path.join(args.path, args.speech_type, "Amazon+Google_voices/test/wav", args.gender))
	EER, minDCF = s.eval_network(eval_list = os.path.join(args.path, args.speech_type, "Amazon+Google_voices", args.speech_type + "_" + "eval_list.txt"), eval_path = os.path.join(args.path, args.speech_type, "Amazon+Google_voices/test/wav"))
	print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = ECAPAModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = ECAPAModel(**vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = ECAPAModel(**vars(args))

EERs = []
score_file = open(args.score_save_path, "a+")

while(1):

	##leave one speaker out cross validation
	#for speaker_to_exclude in all_speakers:
	
	## Training for one epoch
	#print("trainLoader content 3:", len(trainLoader))
	loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader)
	writer.add_scalars('Loss/train', {'loss': loss}, epoch)
	writer.add_scalars('Accuracy/train', {'accuracy': acc}, epoch)

	## Evaluation every [test_step] epochs
	if epoch % args.test_step == 0:
		s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
		#EERs.append(s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)[0])
		#EERs.append(s.eval_network(eval_list = os.path.join(args.path, args.speech_type, "Amazon+Google_voices", args.speech_type + "_" + args.gender + "_eval_list.txt"), eval_path = os.path.join(args.path, args.speech_type, "Amazon+Google_voices/test/wav", args.gender))[0])
		EERs.append(s.eval_network(eval_list = os.path.join(args.path, args.speech_type, "Amazon+Google_voices", args.gender + "_" + "eval_list.txt"), eval_path = os.path.join(args.path, args.speech_type, "Amazon+Google_voices/test/wav", args.gender))[0])
		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, EERs[-1], min(EERs)))
		score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs)))
		score_file.flush()

	if epoch >= args.max_epoch:
		writer.close()
		quit()

	epoch += 1
