# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Global variables
port_number = 5000
inputted, firstinp, secondinp = [], [], []
win = 0
played = [], [], [], [], [], 0, [], [], [], []

# Helper functions from your original code
def prepare_data(sequence, n_lags=2):
    X, y = [], []
    for i in range(len(sequence) - n_lags):
        X.append(sequence[i:i + n_lags])
        y.append(sequence[i + n_lags])
    return np.array(X), np.array(y)

def predict_next(sequence, n_lags=2):
    if len(sequence) < n_lags + 1:
        raise ValueError("short")
    X, y = prepare_data(sequence, n_lags)
    if X.size == 0 or y.size == 0:
        raise ValueError("short")
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    last_values = np.array(sequence[-n_lags:]).reshape(1, -1)
    next_number = model.predict(last_values)
    return next_number

def normal_pdf(x, mean, sigma):
    factor = 1 / (sigma * (2 * 3.141592653589793)**0.5)
    exponent = -((x - mean)**2) / (2 * sigma**2)
    return factor * (2.718281828459045**exponent)

def normaldist(target_first_digit, target_second_digit, weight):
    global confidence
    for key in confidence.keys():
        if key != "100": first_digit = int(key[0])
        else: first_digit = 10
        distance = abs(first_digit - target_first_digit)
        confidence[key] += (normal_pdf(distance, 0, 2)) * weight
        second_digit = int(key[1])
        distance = abs(second_digit - target_second_digit)
        confidence[key] += (2 - (distance / 10)) * weight
        if int(key) == (target_first_digit * 10 + target_second_digit):
            confidence[key] += 1 * weight
        if int(key[1]) == (target_second_digit):
            confidence[key] += 0.5 * weight
    return confidence

def build_markov_chain(data, k):
    markov_chain = {}
    for i in range(len(data) - k):
        current_state = tuple(data[i:i+k])
        next_state = data[i + k]
        if current_state not in markov_chain:
            markov_chain[current_state] = {}
        if next_state not in markov_chain[current_state]:
            markov_chain[current_state][next_state] = 0
        markov_chain[current_state][next_state] += 1
    return markov_chain

def predict_next_elementmark(markov_chain, current_state):
    while current_state not in markov_chain and len(current_state) > 1:
        current_state = current_state[1:]
    if current_state in markov_chain:
        transitions = markov_chain[current_state]
        total_count = sum(transitions.values())
        if total_count > 0:
            probabilities = {state: count / total_count for state, count in transitions.items()}
            next_state = max(probabilities, key=probabilities.get)
            return next_state
    overall_transitions = {}
    for state, transitions in markov_chain.items():
        for next_state, count in transitions.items():
            overall_transitions[next_state] = overall_transitions.get(next_state, 0) + count
    if overall_transitions:
        total_count = sum(overall_transitions.values())
        if total_count > 0:
            probabilities = {state: count / total_count for state, count in overall_transitions.items()}
            next_state = max(probabilities, key=probabilities.get)
            return next_state
    return None

def othernormaldist(target_number, weight):
    global confidence
    for key in confidence.keys():
        number = int(key)
        distance = abs(number - target_number)
        confidence[key] += (10 * normal_pdf(distance, 0, 10)) * weight
    for key in confidence.keys():
        number = int(key)
        if number == target_number:
            confidence[key] += 0.35 * weight

def differencepred():
    global nextfirstdiff, nextseconddiff, confidence, firstinp, secondinp, inputted
    confidence = {str(i).zfill(2): 0 for i in range(0, 101)}
    if len(inputted) == 0: return confidence
    try:
        if inputted[-1] == "100": firstinp.append(10)
        #adds to the first digit inputted list, 100's first digit gets treated as a 10
        else: firstinp.append(int(inputted[-1][0]))
        secondinp.append(int(inputted[-1][1]))
        #adds to the second digit inputted list
    except: pass
    nextfirstdiff, nextseconddiff = None, None
    train = firstdataset + firstinp
    #predict_next function is random forrest
    try: nextfirstdiff = round(float(predict_next(train)))
    except ValueError: pass
    if nextfirstdiff == 10: nextseconddiff = 0
    else:
        train = seconddataset + secondinp
        try: nextseconddiff = round(float(predict_next(train)))
        except ValueError: pass
    if nextseconddiff and nextfirstdiff: normaldist(nextfirstdiff, nextseconddiff, 1) #1 is the weight, higher weight rewards what this regressor chose to be more valued
    nextfirstdiff, nextseconddiff = None, None
    try:
        nextfirstdiff = frequency[inputted[-1]][0]
        if nextfirstdiff == 10: nextseconddiff = 0
        else: nextseconddiff = frequency[inputted[-1]][1]
        normaldist(nextfirstdiff, nextseconddiff, 1.1)
    except: pass
    nextfirstdiff, nextseconddiff = None, None
    train = firstdataset + firstinp
    try:
        markov_chain = build_markov_chain(train, 1)
        current_state = tuple(train[-1:])
        nextfirstdiff = int(predict_next_elementmark(markov_chain, current_state))
    except: pass
    if nextfirstdiff == 10: nextseconddiff = 0
    else:
        try:
            train = seconddataset + secondinp
            markov_chain = build_markov_chain(train, 1)
            current_state = tuple(train[-1:])
            nextseconddiff = int(predict_next_elementmark(markov_chain, current_state))
        except: pass
    if nextseconddiff and nextfirstdiff: normaldist(nextfirstdiff, nextseconddiff, 1.7)
    nextfirstdiff, nextseconddiff = None, None
    #this is xgb
    try:
        X_train = []
        y_train = []
        window_size = 10
        for i in range(len(firstinp) - window_size):
            group = firstinp[i:i+window_size]
            mean = np.mean(group)
            std_dev = np.std(group)
            median = np.median(group)
            max_val = np.max(group)
            min_val = np.min(group)
            range_val = max_val - min_val
            X_train.append([mean, std_dev, median, max_val, min_val, range_val])
            y_train.append(firstinp[i+window_size])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        model = xgb.XGBRegressor(n_estimators=35, max_depth=10, learning_rate=0.11, objective='reg:squarederror')
        model.fit(X_train, y_train)
        next_group = firstinp[-window_size:]
        mean = np.mean(next_group)
        std_dev = np.std(next_group)
        median = np.median(next_group)
        max_val = np.max(next_group)
        min_val = np.min(next_group)
        range_val = max_val - min_val
        nextfirstdiff = int(model.predict(np.array([[mean, std_dev, median, max_val, min_val, range_val]])))
    except: pass
    if nextfirstdiff == 100: nextseconddiff = 0
    else:
        try:
            X_train = []
            y_train = []
            window_size = 10
            for i in range(len(secondinp) - window_size):
                group = secondinp[i:i+window_size]
                mean = np.mean(group)
                std_dev = np.std(group)
                median = np.median(group)
                max_val = np.max(group)
                min_val = np.min(group)
                range_val = max_val - min_val
                X_train.append([mean, std_dev, median, max_val, min_val, range_val])
                y_train.append(secondinp[i+window_size])
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            model = xgb.XGBRegressor(n_estimators=35, max_depth=10, learning_rate=0.11, objective='reg:squarederror')
            model.fit(X_train, y_train)
            next_group = secondinp[-window_size:]
            mean = np.mean(next_group)
            std_dev = np.std(next_group)
            median = np.median(next_group)
            max_val = np.max(next_group)
            min_val = np.min(next_group)
            range_val = max_val - min_val
            nextseconddiff = int(model.predict(np.array([[mean, std_dev, median, max_val, min_val, range_val]])))
        except: pass
    if nextseconddiff and nextfirstdiff: normaldist(nextfirstdiff, nextseconddiff, 1.1)
    #everything from here and below is nearly a duplicate of above but for guessing the main number instead of the first and second individually
    train = dataset + inputted
    nextfirstdiff = None
    try: nextfirstdiff = round(float(predict_next(train)))
    except: pass
    if nextseconddiff: othernormaldist(int(nextfirstdiff), 8) #yet again, 8 is the weight
    nextfirstdiff = None
    try:
        markov_chain = build_markov_chain(train, 1)
        current_state = tuple(train[-1:])
        nextfirstdiff = int(predict_next_elementmark(markov_chain, current_state))
    except: pass
    if nextfirstdiff: othernormaldist(int(nextfirstdiff), 4.6)
    nextfirstdiff = None
    try: nextfirstdiff = frequency2[inputted[-1]]
    except: pass
    if nextfirstdiff: othernormaldist(int(nextfirstdiff), 4.8)
    return confidence


def main():
    global inputted, retro, temp, tempc, next_element, confidence, firstinp, secondinp
    next_element, difference = 0, 0
    confidence = differencepred()
    #strict patterns in the dataset can be found, such as pi, e, and just other common tendencies
    for i in range(len(dataset)):
        confidence[dataset[i]] += (20609+len(inputted))/7500000
        try:
            for j in range(2, min(1000002, len(dataset) - i)):
                temp, tempc = [], []
                for k in range(j):
                    temp.insert(0, dataset[i - k])
                    tempc.insert(0, inputted[-1 - k])
                if temp == tempc: confidence[dataset[i + 1]] += (j - 1) * 4.6
                else: break
        except: pass
    #looks for strict patterns the user inputted, like 1 2 1 2 1 2 1 2 -> 1 get's heavally rewarded because of how long this strange pattern has been being inputted for
    for i in range(len(inputted)):
        retro = i / (len(inputted))
        confidence[inputted[i]] += 0.7 * retro
        for j in range(2, min(1000002, len(inputted) - i)):
            temp, tempc = [], []
            for k in range(j):
                temp.insert(0, inputted[i - k])
                tempc.insert(0, inputted[-1 - k])
            if temp == tempc: confidence[inputted[i + 1]] += (j - 1) * 10.9 * retro
            else: break
    #arithmetic predictor that is weak but only requires 2 prior numbers to be in this set {1, 2, 3, 5, 10, 20, -1, -2, -3, -5, -10, -20} before allowing a prediction
    if (len(inputted) >= 2) and (int(inputted[-2]) - int(inputted[-1]) in {1, 2, 3, 5, 10, 20, -1, -2, -3, -5, -10, -20}):
        next_element = int(inputted[-1]) + (int(inputted[-1]) - int(inputted[-2]))
        if (0 <= next_element <= 9): next_element = f"0{next_element}"
        if (0 <= int(next_element) <= 100): confidence[str(next_element)] += 10
    #arithmetic predictor that looks at the past 3 prior numbers and if that have the same difference to boost the difference onto the last inputted number
    if (len(inputted) >= 3) and (inputted[-1] != inputted[-2]) and (int(inputted[-1]) - int(inputted[-2])) == (int(inputted[-2]) - int(inputted[-3])):
        difference = int(inputted[-1]) - int(inputted[-2])
        next_element = int(inputted[-1]) + difference
        if (0 <= next_element <= 9): next_element = f"0{next_element}"
        if (0 <= int(next_element) <= 100): confidence[str(next_element)] += 30
    #weak geometric predictor that looks for multiples of strictly 2 or divisions of 2
    try:
        if (len(inputted) >= 2) and ((int(inputted[-2])/int(inputted[-1])) in {2, 0.5}):
            next_element = int(int(inputted[-1]) * (int(inputted[-1]) / int(inputted[-2])))
            if (0 <= int(next_element) <= 9): next_element = f"0{next_element}"
            if (0 <= int(next_element) <= 100): confidence[str(next_element)] += 7
    except: pass
    #if the past 3 numbers have all had the same ratio to be confident in the next number with that ratio
    try:
        ratios = [int(inputted[i]) / int(inputted[i-1]) for i in range(len(inputted)-3, len(inputted))]
        if all(ratio == ratios[0] for ratio in ratios):
            next_element = int((int(inputted[-1])) * ratios[0])
            if (0 <= next_element <= 9): next_element = f"0{next_element}"
            if (0 <= int(next_element) <= 100): confidence[str(next_element)] += 30
    except: pass
    #this is the default return if there's been nothing entered prior
    if (len(inputted)) == 0: return "37"
    try:
        if (inputted[-1] == played[1]) and (inputted[-2] == played[2]): return played[0]
    except: pass
    #this never ever happens idfk why i put this here ages ago
    if max(confidence.items()) == 0.0: return inputted[-1]
    #inverts confidence and returns most confident
    inverted_confidence = {v: k for k, v in confidence.items()}
    return inverted_confidence[max(confidence.values())]


# Import your dataset variables here
dataset = []  # Add your dataset here
firstdataset = []  # Add your first dataset here
seconddataset = []  # Add your second dataset here
frequency = {}  # Add your frequency dict here
frequency2 = {}  # Add your frequency2 dict here

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global inputted, win
    
    data = request.json
    number = data.get('number')
    
    if not (0 <= int(number) <= 100):
        return jsonify({'error': 'Invalid number'}), 400
        
    # Format number to have leading zero if needed
    if 0 <= int(number) <= 9:
        number = f"0{number}"
        
    # Add to input list
    inputted.append(number)
    
    # Get prediction using your main logic
    prediction = main()  # Implement your main prediction function
    
    # Check if prediction was correct
    correct = (number == prediction)
    if correct:
        win += 1
    
    # Calculate win rate
    win_rate = (win/len(inputted)*100) if inputted else 0
    
    return jsonify({
        'prediction': prediction,
        'correct': correct,
        'winRate': round(win_rate, 3),
        'roundsPlayed': len(inputted)
    })

if __name__ == '__main__':
    app.run(debug=True,port=port_number)