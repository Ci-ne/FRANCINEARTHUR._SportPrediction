import numpy as np
from unittest import result
from flask import Flask, request, jsonify, render_template
import pickle

model = pickle.load(open('modelXGBRegressor.pkl','rb'))
fifapredict = Flask(__name__)

@fifapredict.route('/')
def home():
    return render_template('index.html')

@fifapredict.route('/predict',methods=['POST'])
def predict():
    thefeatures = [float(x) for x in result.form.values()]
    potential = thefeatures['potential']
    wage_eur = thefeatures['wage_eur']
    age = thefeatures['age']
    international_reputation = thefeatures['international_reputation']
    pace = thefeatures['pace']
    defending = thefeatures['defending']

    attackinglist = ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys']
    attack_rate=[thefeatures[key] for key in attackinglist]
    attacking = sum(attack_rate)/len(attack_rate)
    
    skilllist = ['skill_moves', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control']
    skill_rate=[thefeatures[key] for key in skilllist]
    skill = sum(skill_rate)/len(skill_rate)
    
    movementlist = [ 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance']
    movement_rate=[thefeatures[key] for key in movementlist]
    movement = sum(movement_rate)/len(movement_rate)
    
    powerlist = ['power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots']
    power_rate=[thefeatures[key] for key in powerlist]
    power = sum(power_rate)/len(power_rate)
    
    mentalitylist = [ 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure']
    mentality_rate=[thefeatures[key] for key in mentalitylist]
    mentality = sum(mentality_rate)/len(mentality_rate)
    
    defendinglist = ['defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle']
    defending_rate=[thefeatures[key] for key in defendinglist]
    defending = sum(defending_rate)/len(defending_rate)
    
    goalkeepinglist = ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed']
    goalkeeping_rate=[thefeatures[key] for key in goalkeepinglist]
    goalkeeping = sum(goalkeeping_rate)/sum(goalkeeping_rate)

    averagedskills = [attacking,skill,movement,power,mentality,defending,goalkeeping]
    threshold = sum(averagedskills)/len(averagedskills)
    relskills = [skil for skil in averagedskills if skill>threshold]
    relskil = sum(relskills)/len(relskills)

    posstats = ['ls','st','rs','lw','lf','cf','rf','rw','lam','cam','ram','lm','lcm','cm','rcm','rm','lwb','ldm','cdm','rdm','rwb','lb','lcb','cb','rcb','rb','gk']
    postatvalue = [thefeatures[stat] for stat in posstats]
    relrat = sum(postatvalue)/len(postatvalue)

    features=[potential,wage_eur,age,international_reputation,pace,defending,relskil,relrat]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('index.html', prediction_text='The player"s overall rating {}'.format(output))


if __name__ == "__main__":
    fifapredict.run(debug=True)
