import sys
import json
import shutil
from model_generation import Solver, Model, ActiveGrid, \
                             ModelTime, DataSource, ModelBoundary, \
                             VectorSource, ModelLayer, PropSource


class ModflowModel(object):
    """ """

    def __init__(self, model_data):

        # Vector input data source
        self.vector_source = VectorSource(
            model_data['xmin'], model_data['ymin'],
            model_data['xmax'], model_data['ymax'],
            model_data['n_points'], model_data['n_dim']
            )
        # Time-series input data source
        self.data_source = DataSource(
            model_data['nper'],
            model_data['b_types']
        )
        self.data_source.generate_data()
        # Layer properties data source
        self.properties_source = PropSource(
            model_data['layer_props_params'], self.vector_source.random_points
        )
        self.properties_source.set_params_data()
        # Model grid
        self.active_grid = ActiveGrid(
            model_data['xmin'], model_data['ymin'],
            model_data['xmax'], model_data['ymax'],
            model_data['nx'], model_data['ny'], model_data['nlay']
            )
        self.active_grid.set_ibound(self.vector_source.polygon)
        # Model time
        self.model_time = ModelTime(
            model_data['nper'], model_data['perlen'],
            model_data['nstp'], model_data['steady']
            )
        # Model boundary
        self.model_boundary = ModelBoundary(
            self.active_grid, self.model_time, self.data_source
            )
        self.model_boundary.set_line_segments(self.vector_source.polygon)
        self.model_boundary.set_well(self.vector_source.inner_points)
        self.model_boundary.set_boundaries_spd()

        # Model layer
        self.model_layer = ModelLayer(
            self.properties_source, self.active_grid
        )
        self.model_layer.set_properties()
        self.model_layer.reshape_properties()
        # Model solver
        self.model_solver = Solver(
            model_data['headtol'],
            model_data['maxiterout'],
            model_data['mxiter'],
            model_data['iter1'],
            model_data['hclose'],
            model_data['rclose']
            )
        # Flopy model
        self.model = Model(
            model_name=model_data['model_name'],
            workspace=model_data['workspace'],
            version=model_data['version'],
            exe_name=model_data['exe_name'],
            verbose=model_data['verbose'],
            model_solver=self.model_solver,
            model_time=self.model_time,
            model_grid=self.active_grid,
            model_boundary=self.model_boundary,
            model_layer=self.model_layer,
        )
        # Flopy packages
        self.mf = self.model.get_mf()
        self.pcg = self.model.get_pcg(self.mf)
        self.dis = self.model.get_dis(self.mf)
        self.bas = self.model.get_bas(self.mf)
        self.lpf = self.model.get_lpf(self.mf)
        self.chd = self.model.get_chd(self.mf)
        self.riv = self.model.get_riv(self.mf)
        self.wel = self.model.get_wel(self.mf)
        self.oc = self.model.get_oc(self.mf)
        self.mt = None

        self.lmt = self.model.get_lmt(self.mf)
        self.mt = self.model.get_mt(self.mf)
        self.btn = self.model.get_btn(self.mt)
        self.adv = self.model.get_adv(self.mt)
        self.gcg = self.model.get_gcg(self.mt)
        self.ssm = self.model.get_ssm(self.mt, model_data['concentration'], model_data['itype'])

    def write_files(self):
        self.mf.write_input()
        if self.mt:
            self.mt.write_input()

    def run_model(self):
        success, output = self.mf.run_model(
            silent=False,
            pause=False,
            report=True
            )
        for i in output:
            print(i)
        self.mt.run_model()
        return success

def main(workspace, model_name, nper):
    """ """
    model_data = {
        'xmin': 0,
        'xmax': 1000,
        'ymin': 0,
        'ymax': 1000,
        'nlay': 1,
        'nper': nper,
        'perlen': [100] * nper,
        'nstp': [1] * nper,
        'steady': False,
        'nx': 50,
        'ny': 50,
        'n_points': 10,
        'n_dim': 2,
        'b_types': {
            'NFL':{},
            'CHD': {'min': 100, 'max': 120,
                    'periods': [2, 3]},
            'RIV': {'min': 80, 'max': 90,
                    'periods': [3, 4]},
            'WEL': {'min': 1000, 'max': 5000,
                    'periods': [2, 3]},
            },
        'layer_props_params': {
            'hk': {'min': 0.1, 'max': 10.},
            'hani': {'min': 0.5, 'max': 2},
            'vka': {'min': 1, 'max': 10},
            'top': {'min': 100, 'max': 150},
            'botm': {'min': 0, 'max': 50},
            },
        'concentration': 10,
        'itype': 2,
        'headtol': 0.01,
        'maxiterout': 100,
        'mxiter': 100,
        'iter1': 100,
        'hclose': 0.1,
        'rclose': 0.1,
        'model_name': model_name,
        'workspace': workspace,
        'version': 'mf2005',
        'exe_name': 'mf2005',
        'verbose': True
    }

    try:
        shutil.rmtree(model_data['workspace'])
        print('Updating model in ', model_data['workspace'])
        print('Old model data deleted')
    except FileNotFoundError:
        print('Writing new model to ', model_data['workspace'])

    model = ModflowModel(model_data)
    model.write_files()
    success = model.run_model()
    return success

if __name__ == '__main__':
    number_of_models = sys.argv[1]

    models = {'modelNames': []}

    for i in range(int(number_of_models)):
        model_name = 'model_' + str(i)
        workspace = 'models\\' + 'model_' + str(i)
        nper = 5
        success = main(workspace, model_name, nper)
        if success:
            models['modelNames'].append(model_name)

    with open('models\\models.json', 'w') as f:
        json.dump(models, f)
