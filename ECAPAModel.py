'''
This part is used to train the speaker model and evaluate the performances
'''

import random
import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
		super(ECAPAModel, self).__init__()
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).cuda()

		## HERE FREEZING THE PARAMETER FOR OTHER LAYERS, EXCEPT FC LAYER
		print("Start freezing layers...")
		for name, param in self.speaker_encoder.named_parameters():
			if "fc6" not in name:
				param.requires_grad = False
		print("Complete freezing...")

		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda()

		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		print("entering loop, checking parameter...")
		if len(loader) == 0:
			print("Loader is empty")
		else:
			for num, (data, labels) in enumerate(loader, start = 1):
				print("inside loop")
				self.zero_grad()
				labels            = torch.LongTensor(labels).cuda()

				print("This is the labels in ECAPAModel:", labels)
				#print("This is the data in ECAPAModel:", data)
				#print("This is data shape:", data.shape)

				speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug = True)

				## checking speaker_embedding
				#print("Speaker Embeddding Shape:", speaker_embedding.shape)
				#print("Speaker Embedding Content:")
				#print(speaker_embedding)

				#print("Labels:", labels)

				nloss, prec, probabilities     = self.speaker_loss.forward(speaker_embedding, labels)			
				nloss.backward()
				self.optim.step()

				predicted_labels = torch.argmax(probabilities, dim=1)
				print("predicted labels:", predicted_labels)

				index += len(labels)
				top1 += prec
				loss += nloss.detach().cpu().numpy()
				sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
				" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
				" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
				sys.stderr.flush()
				#print("exiting loop")
			sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)

	def eval_network(self, eval_list, eval_path):
		self.eval()
		files = []
		#embeddings = {}

		#predicted_emotions ={}
		true_label = []

		scores, labels  = [], []

		lines = open(eval_list).read().splitlines()
		#print("This is lines in eval_network", lines)


		for line in lines:
			files.append(line.split()[1])
			#files.append(line.split()[2])
			labels.append(int(line.split()[0]))
			#true_emotion_labels.append(true_label)
		
		#print(files)
		#print(labels)

		# Create a dictionary mapping each filename to its index in the original list
		index_dict = {filename: index for index, filename in enumerate(files)}

		setfiles = list(set(files))
		setfiles.sort(key=lambda x: index_dict[x])

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			print(file)
			print("True label in evluation:", labels[idx])
			#print(torch.LongTensor(labels[idx]).cuda())
			# Full utterance
			#data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)[0]).cuda()
			#data = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			#feats = []
			#startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			startframe = numpy.int64(random.random()*(audio.shape[0]-max_audio))
			#print("startframe in evaluation:", startframe)
			#for asf in startframe:
			#	feats.append(audio[int(asf):int(asf)+max_audio])
			audio = audio[startframe:startframe + max_audio]
			#feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			feat = numpy.stack([audio],axis=0)
			data_2 = torch.FloatTensor(feat[0]).cuda()
			new_data_2 = torch.unsqueeze(data_2, 0)
			
			#print(new_data_2.size())

			# Speaker embeddings
			with torch.no_grad():
				#embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				#embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(new_data_2, aug = False)
				#embedding_2 = F.normalize(embedding_2, p=2, dim=1)
				#print(embedding_2)
				#print("true_label", labels)
				_, _, probabilities_eval = self.speaker_loss.forward(embedding_2, labels[idx])
				max_prob, predicted_emotion_eval = torch.max(probabilities_eval, dim=1)

			print("calculated scores in evaluation:", probabilities_eval)
			print("predicted label in evaluation:", predicted_emotion_eval)	
			#embeddings[file] = [embedding_1, embedding_2]
			scores.append(max_prob.cpu())
		#labels.append(labels)

		"""
		for line in lines:			
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))
		"""
		print("calculated scores:", scores)

		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		return EER, minDCF

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			#print("This is name %s and origname %s in for loop:"%(name, origname))
			#print("This is the size comparison of name %s and origname %s"%(self_state[name].size(), loaded_state[origname].size()))
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
				
			#print("This is name %s and origname %s:"%(name, origname))


			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)