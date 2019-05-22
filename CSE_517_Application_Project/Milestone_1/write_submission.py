import csv

def write_submission(testID, preds):
    f = 'submission.csv'
    with open(f, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        row = ['ID', 'Horizontal_Distance_To_Fire_Points']
        csvwriter.writerow(row)
        for i in range(len(preds)):
            csvwriter.writerow([str(testID[i]), str(preds[i])])
