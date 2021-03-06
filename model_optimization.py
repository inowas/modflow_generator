import random
import numpy as np
import os
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import flopy
from utils_model_optimization import GhostWell, prepare_packages, drop_iface


"""
This code is an implementation of a Genetic optimization algorithm
(https://en.wikipedia.org/wiki/Genetic_algorithm).
Python DEAP library is used (https://github.com/DEAP).
Position of an injection well (lay, row, col) is beeing optimized.
Objecive function - maximize storage increase with respect to reference situation.
"""


class ModflowOptimization(object):
    """
    Modflow optimization class
    """
    def __init__(self, data):
        self.request_data = data
        self.model_original = flopy.modflow.Modflow.load(
            f=data['model_name'],
            model_ws=data['workspace']
            )
        self.model_updated = None
        self.stress_periods = data['time']['stress_periods']
        self.steady = data['time']['steady']
        self.ngen = data['ngen']
        self.pop_size = data['pop_size']
        self.mutpb = data['mutpb']
        self.cxpb = data['cxpb']
        self.control_layer = data['control_layer']
        self.ghost_wells = [
            GhostWell(idx, well_data) for idx, well_data in enumerate(data['wells'])
            ]

        self.variables_map = {}
        self.reference_head_mean = None

    def initialize(self):
        """ Model update. Set reference head value """
        # Set model object with new stress periods
        self.model_updated = prepare_packages(
            self.model_original,
            self.stress_periods
            )
        self.model_updated.write_input()
        self.model_updated.run_model()
        # Set reference head value
        head_file_name = os.path.join(self.model_updated.model_ws, self.model_updated.name)
        head_file_object = flopy.utils.HeadFile(head_file_name + '.hds')

        reference_head = head_file_object.get_alldata(
            mflay=self.control_layer,
            nodata=-9999
            )

        self.reference_head_mean = np.mean(reference_head, axis=0)

        head_file_object.close()

    def generate_candidate(self):
        """Generate initial candidate and
           map candidates variables into variables_map list """
        candidate = []
        for well in self.ghost_wells:
            self.variables_map[well.idx] = {}
            if 'lay' in well.well_variables:
                candidate.append(random.randint(well.constrains['layer_min'],
                                                well.constrains['layer_max']))
                self.variables_map[well.idx]['lay'] = len(candidate) - 1
            if 'row' in well.well_variables:
                candidate.append(random.randint(well.constrains['row_min'],
                                                well.constrains['row_max']))
                self.variables_map[well.idx]['row'] = len(candidate) - 1
            if 'col' in well.well_variables:
                candidate.append(random.randint(well.constrains['col_min'],
                                                well.constrains['col_max']))
                self.variables_map[well.idx]['col'] = len(candidate) - 1

        print(' '.join(['INITIAL CANDIDATE WELL:', str(candidate)]))
        return candidate

    def mutate(self, individual):
        """ Mutation of an individual """
        print('MUTATION_IN: ' + str(individual))
        def mean_(min_, max_):
            return (min_ + max_) / 2

        def std_(min_, max_):
            return (max_ - ((min_ + max_) / 2)) / 3

        i = 0
        for well_key in self.variables_map:
            for variable_key in self.variables_map[well_key]:
                if variable_key == 'lay':
                    individual[i] = random.randint(
                        self.ghost_wells[well_key].constrains['layer_min'],
                        self.ghost_wells[well_key].constrains['layer_max']
                        )
                elif variable_key == 'row':
                    individual[i] = int(
                        random.gauss(
                            mean_(self.ghost_wells[well_key].constrains['row_min'],
                                  self.ghost_wells[well_key].constrains['row_max']),
                            std_(self.ghost_wells[well_key].constrains['row_min'],
                                 self.ghost_wells[well_key].constrains['row_max']),
                        )
                    )
                elif variable_key == 'col':
                    individual[i] = int(
                        random.gauss(
                            mean_(self.ghost_wells[well_key].constrains['col_min'],
                                  self.ghost_wells[well_key].constrains['col_max']),
                            std_(self.ghost_wells[well_key].constrains['col_min'],
                                 self.ghost_wells[well_key].constrains['col_max']),
                        )
                    )
                i += 1
        print('MUTATION_OUT: ' + str(individual))
        return individual,

    def update_well_package(self, individual):
        """Update WEL package"""
        if 'WEL' in self.model_updated.get_package_list():
            wel_package = self.model_updated.get_package('WEL')
            spd = wel_package.stress_period_data.data

            for well in self.ghost_wells:
                spd = well.append_to_spd(
                    spd=spd,
                    individual=individual,
                    variables_map=self.variables_map
                    )
            # remove old wel package
            self.model_updated.remove_package('WEL')
            # write new wel package
            wel_new = flopy.modflow.ModflowWel(
                self.model_updated,
                stress_period_data=spd
                )
            wel_new.write_file()
        # Or write new package
        else:
            spd = {
                key: None for key in self.request_data['time']['stress_periods']
                }
            for well in self.ghost_wells:
                spd = well.append_to_spd(
                    spd=spd,
                    individual=individual,
                    variables_map=self.variables_map
                    )
            # remove old wel package
            self.model_updated.remove_package('WEL')
            # write new wel package
            wel_new = flopy.modflow.ModflowWel(
                self.model_updated,
                stress_period_data=spd
                )
            wel_new.write_file()

    def evaluate(self, individual):
        """ Add well, ran model and evaluate results """
        self.update_well_package(individual)
        # Run model
        silent = True
        pause = False
        report = False

        success, buff = self.model_updated.run_model(silent, pause, report)
        # Read results
        if success:
            head_file_object = flopy.utils.HeadFile(
                os.path.join(
                    self.model_updated.model_ws,
                    self.model_updated.name + '.hds'
                    )
                )
            new_reference_head = head_file_object.get_alldata(
                mflay=self.control_layer,
                nodata=-9999
                )
            new_reference_head_mean = np.mean(new_reference_head, axis=0)

            fitness = np.nanmean(new_reference_head_mean - self.reference_head_mean)
        else:
            fitness = -9999

        return fitness,

    def optimize_model(self):
        """
        DEAP Optimization
        """
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("candidate", self.generate_candidate)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.candidate)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mutate", self.mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=self.pop_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("mean", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        self.hall_of_fame = tools.HallOfFame(maxsize=100)

        self.result, self.log = algorithms.eaSimple(
            population, toolbox,
            cxpb=self.cxpb, mutpb=self.mutpb,
            ngen=self.ngen, stats=stats,
            halloffame=self.hall_of_fame,
            verbose=False
            )
        print(self.result)
        print(self.log)
        return self.hall_of_fame
