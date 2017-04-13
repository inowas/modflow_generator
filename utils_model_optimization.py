import flopy

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
