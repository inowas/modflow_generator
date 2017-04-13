import random
import numpy as np
import os
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import flopy
from utils_model_optimization import prepare_packages, drop_iface


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
        self.model_updated
        self.stress_periods = data['time']['stress_periods']
        self.steady = data['time']['steady']
        self.ngen = data['ngen']
        self.popsize = data['popsize']
        self.control_layer = data['control_layer']
        self.wells_bbox = data['wells_bbox']
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
                self.variables_map[well.idx]['lay'] = len(candidate)
            elif 'row' in well.well_variables:
                candidate.append(random.randint(well.constrains['row_min'],
                                                well.constrains['row_max']))
                self.variables_map[well.idx]['row'] = len(candidate)
            elif 'col' in well.well_variables:
                candidate.append(random.randint(well.constrains['col_min'],
                                                well.constrains['col_max']))
                self.variables_map[well.idx]['col'] = len(candidate)

        print(' '.join(['INITIAL CANDIDATE WELL:', str(candidate)]))
        return candidate

    def mutate(self, individual):
        """ Mutation of an individual """

        for i, variable_idx in enumerate(self.variables_map):
            if variable_idx[0] == 'lay':
                individual[i] = random.randint(
                    self.ghost_wells[variable_idx[1]].constrains['layer_min'],
                    self.ghost_wells[variable_idx[1]].constrains['layer_min']
                    )
            elif variable_idx[0] == 'row':
                individual[i] = random.randint(
                    self.ghost_wells[variable_idx[1]].constrains['row_min'],
                    self.ghost_wells[variable_idx[1]].constrains['row_min']
                    )
            elif variable_idx[0] == 'col':
                individual[i] = random.randint(
                    self.ghost_wells[variable_idx[1]].constrains['col_min'],
                    self.ghost_wells[variable_idx[1]].constrains['col_min']
                    )

        return individual,

    def evaluate(self, individual):
        """ Add well, ran model and evaluate results """
        # Update WEL package
        if 'WEL' in self.model_updated.get_package_list():
            wel_package = self.model_updated.get_package('WEL')
            spd = wel_package.stress_period_data.data

            for well in self.ghost_wells:
                spd = well.append_to_spd(
                    spd_old=spd,
                    individual=individual,
                    variables_map=self.variables_map
                    )

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
                    spd_old=spd,
                    individual=individual,
                    variables_map=self.variables_map
                    )

            wel_new = flopy.modflow.ModflowWel(
                self.model_updated,
                stress_period_data=spd
                )
            wel_new.write_file()

        # Run model
        silent = True
        pause = False
        report = False

        success, buff = self.model.run_model(silent, pause, report)
        # Read results
        if success:
            head_file_objects = flopy.utils.HeadFile(
                os.path.join(
                    self.model.model_ws,
                    self.model.name + '.hds'
                    )
                )
            heads_timestep = head_file_objects.get_data(kstpkper=(0, 0))[-1]
            fitness = np.mean(heads_timestep - self.reference_head_mean),
        else:
            fitness = -9999,

        return fitness

    def optimize_model(self, ngen=30, cxpb=0.5, mutpb=0.1, pop_size=30):
        """
        DEAP Optimization
        """
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register(
            "candidate",
            self.generate_candidate
            )
        toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            toolbox.candidate
            )
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual
            )
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mutate", self.mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=pop_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("mean", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        self.hall_of_fame = tools.HallOfFame(maxsize=100)

        self.result, self.log = algorithms.eaSimple(
            population, toolbox,
            cxpb=cxpb, mutpb=mutpb,
            ngen=ngen, stats=stats,
            halloffame=self.hall_of_fame, verbose=False
            )
        return self.hall_of_fame


class GhostWell(object):
    """Well used in optimization process"""
    def __init__(self,i dx, data):
        self.idx = idx
        self.data = data
        # Area bounding box
        self.constrains = data['constrains']
        self.row_in_spd = None
        self.once_appended = False

        # Variables to be optimized
        self.wel_variables = []
        if 'lay' not in data['location'] or data['location']['lay'] is None:
            self.wel_variables.append('lay')
        if 'row' not in data['location'] or data['location']['row'] is None:
            self.wel_variables.append('row')
        if 'col' not in data['location'] or data['location']['col'] is None:
            self.wel_variables.append('col')

    def append_to_spd(self, spd_old, individual, variables_map):
        """Add candidate well data to SPD """
        # Define lay, row, col
        if 'lay' in variable_map[self.idx]:
            lay = individual[variable_map[self.idx]['lay']]
        else:
            lay = self.data['location']['lay']
        if 'row' in variable_map[self.idx]:
            lay = individual[variable_map[self.idx]['row']]
        else:
            lay = self.data['location']['row']
        if 'col' in variable_map[self.idx]:
            lay = individual[variable_map[self.idx]['col']]
        else:
            lay = self.data['location']['col']

        # Replace previousely appended ghost well with a new one
        if self.once_appended:
            for period in data['pumping']['rates']:
                np.put(
                    spd[period],
                    self.row_in_spd,
                    ([
                        lay, row, col, self.data['pumping']['rates'][period]
                    ])
                    )

        else:
            # Initially append a ghost well
            if spd[period] is None:
                spd[period] = np.recarray()
            for period in data['pumping']['rates']:
                spd[period] = np.append(
                    spd[period],
                    np.array([
                        lay, row, col, self.data['pumping']['rates'][period]
                        ],
                        dtype=spd.dtype)
                    ).view(np.recarray)

                self.row_in_spd = spd[period].shape()[0]
                self.once_appended = True
        return spd
