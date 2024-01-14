import pandas as pd
from preprocess_data import preprocess_data

## Assuming your CSV file is named 'your_data.csv'
#df = pd.read_csv('/home/ray/Abschlussarbeit/ECAPA-TDNN/dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Subjectivedata/neutral_ratings.csv')

## data grouping by speaker's gender and normalization



## labeling with one hot encoded 

#df['warmth'] = ((df['sympathetic_neutral'] >= 50) & (df['kind_neutral'] >= 50)).astype(int)
#print((df['sympathetic_neutral'] >= 50) & (df['kind_neutral'] >= 50))
#df['highly competence'] = ((df['responsible_neutral'] >= 50) & (df['skillful_neutral'] >= 50)).astype(int)

## Display the updated DataFrame
#print(df)
#df.to_csv('/home/ray/Abschlussarbeit/ECAPA-TDNN/dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Subjectivedata/neutral_ratings.csv', index=False)

class labeling:

    def __init__(self):
        self.pre = preprocess_data("/home/ray/Abschlussarbeit/ECAPA-TDNN/dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Subjectivedata/neutral_ratings.csv")
        self.means = self.pre.group_norm_means()
        self.dict = {'00': 0, '01': 1, '10': 2, '11': 3}
        self.result = {}

    def create_label(self):
        print(self.means)
        for speaker in self.means:
            print(speaker)
            print(self.means[speaker])
            label = str(int((self.means[speaker][0] >= 0.5) & (self.means[speaker][1] >= 0.5))) + str(int((self.means[speaker][2] >= 0.5) & (self.means[speaker][3] >= 0.5)))
            self.result[speaker] = label

        return self.result

    def create_label_list(self, labels):
        file_path = "/home/ray/Abschlussarbeit/ECAPA-TDNN/dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data/Neutral_speech/Amazon+Google_voices/train_list_one_hot.txt"

        ## wrtiing to file
        with open(file_path, "w") as file:
            for speaker, label in labels.items():
                if speaker.split('_')[0] == 'Wavenet':
                    formatted_string = str(self.dict[label]) + ' ' + 'Wavenet_' + speaker.split('_')[1] + '/' + speaker + '\n'
                else:
                    formatted_string = str(self.dict[label]) + ' ' + speaker.split('_')[0] + '/' + speaker + '\n'
                file.write(formatted_string)
        
        print(f"File '{file_path}' created successfully.")

if __name__ == "__main__":
    
    res = labeling()
    #print(res.create_label())
    labels = res.create_label()
    res.create_label_list(labels)

##print(means)