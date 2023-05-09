"""
Strong Field Magnetoconvection

Usage:
    bhw_d3.py <run_file>
"""

from pathlib import Path
from configparser import ConfigParser
import sys
import os
from docopt import docopt

import numpy as np
from mpi4py import MPI

from dedalus import public as d3
import logging
logger = logging.getLogger(__name__)

data_dir = "scratch/" + sys.argv[0].split('.py')[0]

# parse arguments
args = docopt(__doc__)
parameter_file = args['<run_file>']
logger.info(f"Reading parameter file {parameter_file}")
parameter_file = Path(parameter_file)
runconfig = ConfigParser()
runconfig.read(parameter_file)
params = runconfig['params']
nx = params.getint('nx') # resolution
ny = params.getint('ny') # resolution
Lx = params.getfloat('Lx', 40)
Ly = params.getfloat('Ly', 40)
seed = params.getint('seed')
filter_frac = params.getfloat('filter', 0.5)

run_opts = runconfig['run']
stop_time = run_opts.getfloat('stop-time', 2)
restart = run_opts.get('restart')

CFL = run_opts.getboolean('CFL')
safety = run_opts.getfloat('safety', 0.1)

inv_zeta = params.getfloat('inv_zeta', 0.5)
R = params.getfloat('R', 2)
inv_sigma_hat = params.getfloat('inv_sigma_hat', 0)

data_dir += f"_{parameter_file.stem}"


if restart:
    if MPI.COMM_WORLD.rank == 0:
        restart_dirs = glob.glob(data_dir+"_restart*")
        if restart_dirs:
            restart_dirs.sort()
            last = int(re.search("_restart(\d+)", restart_dirs[-1]).group(1))
            data_dir += "_restart{}".format(last+1)
        else:
            if os.path.exists(data_dir):
                data_dir += "_restart1"
    else:
        data_dir = None
    data_dir = MPI.COMM_WORLD.bcast(data_dir, root=0)
logger.info("Saving data in {}".format(data_dir))

if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.mkdir('{:s}/'.format(data_dir))

# Create bases and domain
dealias = 3/2
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=np.float64)
ex, ey = coords.unit_vector_fields(dist)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=ny, bounds=(0, Ly), dealias=dealias)

# Fields
psi = dist.Field(name='psi', bases=(xbasis,ybasis))
phi = dist.Field(name='phi', bases=(xbasis,ybasis))
N = dist.Field(name='N')

tau_psi = dist.Field(name='tau_psi')
tau_phi = dist.Field(name='tau_phi')

rho = 1-inv_zeta
u = d3.skew(d3.grad(psi))
# Jacobian
def J(A,B):
    gradA = d3.grad(A)
    gradB = d3.grad(B)
    return ex@gradA * ey@gradB - ey@gradA * ex@gradB

lap2 = lambda A: d3.lap(d3.lap(A))
lap4 = lambda A: d3.lap(d3.lap(d3.lap(d3.lap(A))))
Avg = lambda A: d3.integ(A)/Lx/Ly

problem = d3.IVP([psi, phi, N, tau_psi, tau_phi], time='t', namespace=locals())
logger.info("running with inv_zeta = {}, inv_sigma_hat = {}, R = {}, Lx = {}, Ly = {}".format(inv_zeta, inv_sigma_hat, R, Lx, Ly))

problem.add_equation("rho*dt(lap(phi)) + inv_sigma_hat*lap(dt(lap2(phi))) - lap4(phi) + lap(phi) + R*lap2(phi) + tau_phi = -rho*J(psi, lap(phi)) - inv_sigma_hat*lap(J(psi,lap2(phi))) + N*lap2(phi)")
problem.add_equation("inv_sigma_hat*dt(lap(psi)) - lap2(psi) + tau_psi = -inv_sigma_hat*J(psi,lap(psi)) + 2*J(lap(phi), inv_zeta*phi + inv_sigma_hat*lap2(phi))")
problem.add_equation("dt(N) + 4*N = Avg(grad(lap(phi))@grad(lap(phi)))")

problem.add_equation("integ(psi) = 0")
problem.add_equation("integ(phi) = 0")

# Build solver
solver = problem.build_solver(d3.RK443)
logger.info('Solver built')

# Initial conditions
if restart:
    solver.load_state(restart,-1)
    logger.info("Restarting from time t = {0:10.5e}".format(solver.sim_time))
else:
    logger.info("initializing electron density")
    phi.fill_random('g', seed=seed, distribution='normal', scale=1e-3)
    phi.low_pass_filter(scales=filter_frac)

if CFL:
    CFL = d3.CFL(solver, initial_dt=1e-6, cadence=1, safety=safety,
                         max_change=1.5, min_change=0.5)
    CFL.add_velocity(u)
else:
    dt = 1e-4

# Integration parameters
solver.stop_sim_time = stop_time
solver.stop_wall_time = 60.*60*24*5
solver.stop_iteration = np.inf

# Analysis
check = solver.evaluator.add_file_handler(os.path.join(data_dir,'checkpoints'), wall_dt=3540, max_writes=1)
check.add_tasks(solver.state)

snap = solver.evaluator.add_file_handler(os.path.join(data_dir,'snapshots'), sim_dt=1e-3, max_writes=200)
snap.add_task(psi, name='psi')
snap.add_task(phi, name='phi')
snap.add_task(d3.lap(psi), name='vorticity')

timeseries = solver.evaluator.add_file_handler(os.path.join(data_dir,'timeseries'), sim_dt=1e-3, max_writes=None)
timeseries.add_task(N, name='Nusselt')
timeseries.add_task(tau_phi, name='tau_phi')
timeseries.add_task(tau_psi, name='tau_psi')

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(N, name='Nusselt')

try:
    logger.info('Starting loop')

    while solver.proceed:
        if CFL:
            dt = CFL.compute_timestep()
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info(f"Iteration: {solver.iteration:d}, Time: {solver.sim_time:4.2e}, dt: {dt}")
            logger.info(f"N (min,max) = {flow.min('Nusselt'):4.2e},{flow.max('Nusselt'):4.2e}")
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.evaluate_handlers_now(dt)
    solver.log_stats()


