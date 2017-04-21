import flopy
import numpy as np

class GhostWell(object):
    """Well used in optimization process"""
    def __init__(self, idx, data):
        self.idx = idx
        self.data = data
        # Area bounding box
        self.constrains = data['constrains']
        self.row_in_spd = None
        self.once_appended = False
        # Variables to be optimized
        self.well_variables = []
        if 'lay' not in data['location'] or data['location']['lay'] is None:
            self.well_variables.append('lay')
        if 'row' not in data['location'] or data['location']['row'] is None:
            self.well_variables.append('row')
        if 'col' not in data['location'] or data['location']['col'] is None:
            self.well_variables.append('col')

    def append_to_spd(self, spd, individual, variables_map):
        """Add candidate well data to SPD """
        # Define lay, row, col
        if 'lay' in variables_map[self.idx]:
            lay = individual[variables_map[self.idx]['lay']]
        else:
            lay = self.data['location']['lay']
        if 'row' in variables_map[self.idx]:
            row = individual[variables_map[self.idx]['row']]
        else:
            row = self.data['location']['row']
        if 'col' in variables_map[self.idx]:
            col = individual[variables_map[self.idx]['col']]
        else:
            col = self.data['location']['col']

        # Replace previousely appended ghost well with a new one
        if self.once_appended:
            for period in self.data['flux']:
                np.put(
                    spd[period],
                    self.row_in_spd,
                    ([
                        (lay, row, col, self.data['flux'][period])
                    ])
                    )

        else:
            # Initially append a ghost well
            for period in self.data['flux']:
                if spd[period] is None:
                    spd[period] = np.recarray(
                        0,
                        dtype=[('k', 'i4'), ('i', 'i4'), ('j', 'i4'), ('flux', 'f4')]
                        )
                spd[period] = np.append(
                    spd[period],
                    np.array(
                        [(lay, row, col, self.data['flux'][period])],
                        dtype=spd[period].dtype
                        )
                    ).view(np.recarray)

                self.row_in_spd = len(spd[period]) - 1
                self.once_appended = True

        return spd


def drop_iface(rec):
    """
    Removes 'iface' column from stress period data recarray
    """
    index = rec.dtype.names.index('iface')
    list_ = rec.tolist()
    for row, i in enumerate(list_):
        list_[row] = list(i)
        del list_[row][index]
    return list_


def prepare_packages(model_object, stress_periods):
    """
    Rewrites models spd packages to start/end transient stress_periods
    """

    modflow_spd_packages = {'WEL': flopy.modflow.ModflowWel,
                            'LAK': flopy.modflow.ModflowLak,
                            'RIV': flopy.modflow.ModflowRiv,
                            'CHD': flopy.modflow.ModflowChd,
                            'GHB': flopy.modflow.ModflowGhb}

    print('Reading stress-period-data of the given model object...')
    print(' '.join(
        [
            'Writing new packages for stress periods ',
            str(stress_periods[0]),
            ':',
            str(stress_periods[-1])
        ]
        )
         )

    for package_name in model_object.get_package_list():
        if package_name in modflow_spd_packages:
            print('Preparing SPD for ' + package_name + ' package')
            package = model_object.get_package(package_name)
            spd = {k: v for
                   k, v in package.stress_period_data.data.items()
                   if stress_periods[0] <= k <= stress_periods[-1]}

            if 'iface' in spd[stress_periods[0]].dtype.names:
                print('Removing IFACE from ' + package_name + ' package SPD')
                spd = {k: drop_iface(v) for k, v in spd}

            modflow_spd_packages[package_name] = modflow_spd_packages[package_name](
                model_object,
                stress_period_data=spd
                )

        if package_name == 'DIS':
            print('Preparing DIS package')
            dis = model_object.get_package(package_name)
            perlen = dis.perlen.array[stress_periods[0]:stress_periods[-1] + 1]
            nstp = dis.nstp.array[stress_periods[0]:stress_periods[-1] + 1]
            steady = dis.steady.array[stress_periods[0]:stress_periods[-1] + 1]
            nper = len(perlen)
            delc = dis.delc.array
            delr = dis.delr.array
            nlay = dis.nlay
            nrow = dis.nrow
            ncol = dis.ncol
            top = dis.top.array
            botm = dis.botm.array
            laycbd = dis.laycbd.array
            dis_new = flopy.modflow.ModflowDis(
                model_object, nlay=nlay, nrow=nrow, ncol=ncol,
                delr=delr, delc=delc, top=top, steady=steady,
                botm=botm, laycbd=laycbd, perlen=perlen, nstp=nstp,
                nper=nper
                )

    return model_object
