import sys
import flopy
from model_optimization import ModflowOptimization

def main():
    """ """
    data = {
        'model_name': 'model_1.nam',
        'workspace': 'models\\model_1',
        'ngen': 50,
        'popsize': 30,
        'control_layer': 0,
        'wells': [
            {
                'location': {
                    'lay': 0,
                    },
                'flux': {
                    0: 1000,
                    1: 1000,
                    2: 1000,
                    3: 1000,
                    4: 1000,
                    5: 1000,
                    6: 1000,
                    7: 1000,
                    8: 1000,
                    9: 1000
                    },
                'constrains': {
                    'layer_min': 0,
                    'layer_max': 0,
                    'row_min': 0,
                    'row_max': 100,
                    'col_min': 0,
                    'col_max': 100,
                    'rate_min': -1000,
                    'rate_max': -500
                    }
            },
            {
                'location': {
                    'lay': 0,
                    },
                'flux': {
                    0: 1000,
                    1: 1000,
                    2: 1000,
                    3: 1000,
                    4: 1000,
                    5: 1000,
                    6: 1000,
                    7: 1000,
                    8: 1000,
                    9: 1000
                    },
                'constrains': {
                    'layer_min': 0,
                    'layer_max': 0,
                    'row_min': 0,
                    'row_max': 100,
                    'col_min': 0,
                    'col_max': 100,
                    'rate_min': -1000,
                    'rate_max': -500
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
    print(hall_of_fame)
if __name__ == '__main__':
    main()