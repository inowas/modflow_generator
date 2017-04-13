import sys
import flopy
from model_optimization import ModelOptimization

def main():
    """ """
    data = {
        'model_name': 'tutorial2',
        'workspace': 'C:\\Users\\Notebook\\Documents\\GitHub\\pyprocessing\\optimization\\test_model',
        'ngen': 1,
        'popsize': 1,
        'control_layer': 0,
        'wells': [
            {
                'bbox': {
                    'layer_min': 0,
                    'layer_max': 0,
                    'row_min': 0,
                    'row_max': 100,
                    'col_min': 0,
                    'col_max': 100,
                    'rate_min': 1000,
                    'rate_max': 2000
                    },
                'location': {
                    'lay': 0,
                    'row': 0,
                    'col': 0
                    },
                'pumping': {
                    'rates': {
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
                        }
                    }
                },
            {
                'constrains': {
                    'layer_min': 0,
                    'layer_max': 0,
                    'row_min': 0,
                    'row_max': 100,
                    'col_min': 0,
                    'col_max': 100,
                    'rate_min': -1000,
                    'rate_max': -500
                    },
                'location': {
                    'lay': 0
                    },
                'pumping': {
                    'rates': {
                        0: -1000,
                        1: -1000,
                        2: -1000,
                        3: -1000,
                        4: -1000,
                        5: -1000,
                        6: -1000,
                        7: -1000,
                        8: -1000,
                        9: -1000
                        }
                    }
                }
        ],
        'time': {
            'stress_periods': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'steady': [False, False, False, False, False, False, False, False, False, False]
        }
    }

    MO = ModflowOptimization(data)

if __name__ == '__main__':
    main()