from model.training import TrainingValidation
from util.arguments import args
from util.config import load

pwd = '/home/src/'
args = args()
data_name = args.data
model_name = args.model
data_parameters = load(f'config/data.json', data_name)
model_parameters = load(f'config/model.json', model_name)

training_validation = TrainingValidation(model_name, model_parameters, data_parameters)
training_validation.train()
training_validation.predict()