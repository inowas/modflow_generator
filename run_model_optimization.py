import sys
import json
import flopy
from model_optimization import ModflowOptimization

def main(workspace, model_name):
    """ """
    data = {
        'model_name': model_name + '.nam',
        'workspace': workspace,
        'ngen': 100,
        'pop_size': 10,
        'mutpb': 0.1,
        'cxpb': 0.5,
        'control_layer': 0,
        'wells': [
            {
                'location': {
                    'lay': 0,
                    },
                'flux': {
                    0: 10000,
                    1: 10000,
                    2: 5000,
                    3: 10000,
                    4: 2000,
                    5: 10000,
                    6: 11000,
                    7: 9000,
                    8: 10000,
                    9: 1000
                    },
                'constrains': {
                    'layer_min': 0,
                    'layer_max': 0,
                    'row_min': 0,
                    'row_max': 50,
                    'col_min': 0,
                    'col_max': 50
                    }
            },
            {
                'location': {
                    'lay': 0,
                    },
                'flux': {
                    0: -5000,
                    1: -5000,
                    2: -5000,
                    3: -8000,
                    4: -9000,
                    5: -3000,
                    6: -5000,
                    7: -6000,
                    8: -7000,
                    9: -8000
                    },
                'constrains': {
                    'layer_min': 0,
                    'layer_max': 0,
                    'row_min': 0,
                    'row_max': 50,
                    'col_min': 0,
                    'col_max': 50
                    }
            }
            ],
        'time': {
            'stress_periods': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'steady': [False, False, False, False, False, False, False, False, False, False]
        }
    }

    MO = ModflowOptimization(data)
    MO.initialize()
    hall_of_fame = MO.optimize_model()
    return list(hall_of_fame)



if __name__ == '__main__':
    list_of_models_file = sys.argv[1]

    results = {}
    with open(list_of_models_file, 'r') as f:
        list_of_models = json.load(f)['modelNames']

    for i in list_of_models:
        model_name = i
        workspace = 'models\\' + i
        results[model_name] = main(workspace, model_name)

    with open('models\\optimization-results.json', 'w') as f:
        json.dump(results, f)
