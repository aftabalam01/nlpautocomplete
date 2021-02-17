from pathlib import Path

def measure_accuracy():
    with open(f'{Path(__file__).parent}/../data/answer.txt','r') as f:
        answers = f.readlines()

    with open(f'{Path(__file__).parent}/../output/pred.txt','r') as f:
        preds = f.readlines()
    correct=0
    total = len(preds)
    #assert len(answers)==len(preds)
    for i in range(len(preds)):
        if answers[i].lower() in preds[i].lower():
            correct +=1
    print(F'Number of accurate prediction = {correct} and percentage = {correct*100/total}')

if __name__=='__main__':
    measure_accuracy()