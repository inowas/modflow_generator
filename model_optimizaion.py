import random
import numpy as np
import os
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import flopy
from utils_model_optimization import prepare_packages_steady, drop_iface


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
        self.model = flopy.modflow.Modflow.load(f=data['model_name'],
                                                model_ws=data['workspace'])
        self.stress_periods = data['time']['stress_periods']
        self.steady = data['time']['steady']
        self.ngen = data['ngen']
        self.popsize = data['popsize']
        self.control_layer = data['control_layer']
        self.wells_bbox = data['wells_bbox']
        self.ghost_wells = [GhostWell(well_data) for well_data in data['wells']]

        self.variables_map = []
        self.reference_head_mean = None

    def initialize(self):
        """ Model initialization """
        self.model = prepare_packages_steady(
            self.model,
            self.stress_periods
            )
        self.model.write_input()
        head_file_name = os.path.join(self.model.model_ws, self.model.name)
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
        for idx, well in enumerate(self.ghost_wells):
            if 'lay' in well.well_variables:
                candidate.append(random.randint(well.bbox['layer_min'],
                                                well.bbox['layer_max']))
                self.variables_map.append(('lay', idx))
            if 'row' in well.well_variables:
                candidate.append(random.randint(well.bbox['row_min'],
                                                well.bbox['row_max']))
                self.variables_map.append(('row', idx))
            if 'col' in well.well_variables:
                candidate.append(random.randint(well.bbox['col_min'],
                                                well.bbox['col_max']))
                self.variables_map.append(('col', idx))
            if 'rates' in well.well_variables:
                for i in range(len(self.stress_periods)):
                    candidate.append(random.random(well.bbox['rate_min'],
                                                   well.bbox['rate_max']))
                    self.variables_map.append(('rate', idx))

        print(' '.join(['FIRST CANDIDATE WELL:', str(candidate)]))
        return candidate

    def mutate(self, individual):
        """ Mutation of an individual """

        for i, variable_idx in enumerate(self.variables_map):
            if variable_idx[0] == 'lay':
                individual[i] = random.randint(
                    self.ghost_wells[variable_idx[1]].bbox['layer_min'],
                    self.ghost_wells[variable_idx[1]].bbox['layer_min']
                    )
            elif variable_idx[0] == 'row':
                individual[i] = random.randint(
                    self.ghost_wells[variable_idx[1]].bbox['row_min'],
                    self.ghost_wells[variable_idx[1]].bbox['row_min']
                    )
            elif variable_idx[0] == 'col':
                individual[i] = random.randint(
                    self.ghost_wells[variable_idx[1]].bbox['col_min'],
                    self.ghost_wells[variable_idx[1]].bbox['col_min']
                    )
            elif variable_idx[0] == 'rate':
                individual[i] = random.random(
                    self.ghost_wells[variable_idx[1]].bbox['rate_min'],
                    self.ghost_wells[variable_idx[1]].bbox['rate_min']
                    )

        return individual,

    def evaluate(self, individual):
        """ Add well, ran model and evaluate results """
        # Update WEL package
        if 'WEL' in self.model.get_package_list():
            wel_package = self.model.get_package('WEL')
            spd_old = wel_package.stress_period_data.data

            for idx, well in enumerate(self.ghost_wells):
                spd = well.append_to_spd(spd=spd_old, idx=idx)

            wel_new = flopy.modflow.ModflowWel(self.model,
                                               stress_period_data=spd)
            wel_new.write_file()
        else:
            wel_new = flopy.modflow.ModflowWel(self.model,
                                               stress_period_data={0: [individual]})
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
    def __init__(self, data):
        self.data = data
        # Area bounding box
        self.bbox = data['bbox']
        self.row_in_spd = None
        self.once_appended = self.row_in_spd is not None

        # Variables to be optimized
        self.wel_variables = []
        if 'lay' not in data['location'] or data['location']['lay'] is None:
            self.wel_variables.append('lay')
        if 'row' not in data['location'] or data['location']['row'] is None:
            self.wel_variables.append('row')
        if 'col' not in data['location'] or data['location']['col'] is None:
            self.wel_variables.append('col')
        if 'rates' not in data['pumping'] or data['pumping']['rates'] is None:
            self.wel_variables.append('rates')
        if 'total_volume' not in data['pumping'] or data['pumping']['total_volume'] is None:
            self.wel_variables.append('toltal_volume')

    def append_to_spd(self, well_data, spd):
        """Add candidate well data to SPD """
        # Replace previousely appended ghost well with a new one
        if self.once_appended:
            for period in well_data:
                np.put(spd[period], self.row_in_spd, (well_data[period]))

        else:
            # Initially append a ghost well
            for period in well_data:
                spd[period] = np.append(
                    spd[period],
                    np.array([well_data[period]], dtype=spd.dtype)
                    ).view(np.recarray)
                self.row_in_spd = spd[period].shape()[0]

        return spd
