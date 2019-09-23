import json 

THRESHOLD = 0.95

predictions_fp = 'data/predictions_yesand_bert_classifier8577.json'
with open(predictions_fp, 'r') as f: 
    predictions = json.load(f) 

potential_yesands = [] 
for prediction in predictions: 
    if prediction['confidence']['yesand'] >= THRESHOLD*100: 
        potential_yesands.append(prediction)

proportion = round(len(potential_yesands) / len(predictions) * 100,2) 
print(f"{len(potential_yesands)} predictions, {proportion}% of all predictions, were yes-ands for a confidence threshold of {THRESHOLD}.")

filtered_predictions_fp = predictions_fp[:predictions_fp.find('/')+1] + f'filtered_{THRESHOLD*100}_' + predictions_fp[predictions_fp.find('/')+1:]
with open(filtered_predictions_fp, 'w') as f: 
    json.dump(potential_yesands, f, indent=4) 