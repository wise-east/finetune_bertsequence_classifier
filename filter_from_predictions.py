import json 
import re 

THRESHOLD = 0.95

def replace_bad_characters(text): 

    text = re.sub('\x82', ',', text)
    text = re.sub('\x84', ',,', text)
    text = re.sub('\x85', '...', text)
    text = re.sub('(\x97)+', ' - ', text)
    text = re.sub('(\x96)+', "-", text)
    text = re.sub('(\x95)|(\x99)|(\xa0)', " ", text)
    text = re.sub('(\x92)|(\x91)', "'", text)
    text = re.sub('(\x93)|(\x94)', '"', text)

    return text 

def filter(predictions): 
    ''' 
    Filter predictions based on criteria 
    ''' 
    valid_ending_punctuation = ['.', '!', '?']
    potential_yesands = [] 
    for prediction in predictions: 
        # confidence must be above the threshold 
        if prediction['confidence']['yesand'] < THRESHOLD*100:
            continue 

        # process characters that MTURK cannot process: 
        prediction['p'] = replace_bad_characters(prediction['p'])
        prediction['r'] = replace_bad_characters(prediction['r'])

        # remove instances with ... 
        if '...' in  prediction['p'] or '...' in prediction['r']: 
            continue 
        
        if prediction['p'][-1] not in valid_ending_punctuation or prediction['r'][-1] not in valid_ending_punctuation: 
            continue 

        # remove instances that are shorter than 4 words 
        if len(prediction['p'].split()) < 4 or len(prediction['r'].split()) < 4: 
            continue
        
        # if all criteria are passed, add to potential yes-ands 
        potential_yesands.append(prediction)

    return potential_yesands

predictions_fp = 'data/predictions_yesand_cornell_bert_base_iter1.json'
with open(predictions_fp, 'r') as f: 
    predictions = json.load(f) 

potential_yesands = filter(predictions)

proportion = round(len(potential_yesands) / len(predictions) * 100,2) 
print(f"{len(potential_yesands)} predictions, {proportion}% of all predictions, were yes-ands for a confidence threshold of {THRESHOLD}.")

filtered_predictions_fp = predictions_fp[:predictions_fp.find('/')+1] + f'filtered_{THRESHOLD*100}_' + predictions_fp[predictions_fp.find('/')+1:]
with open(filtered_predictions_fp, 'w') as f: 
    json.dump(potential_yesands, f, indent=4) 