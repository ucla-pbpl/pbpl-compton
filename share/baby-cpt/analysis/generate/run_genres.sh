set -x
mpiexec -np 10 python -m mpi4py 'run-random-image.py'
mpiexec -np 10 python -m mpi4py 'run-random-triag.py'
mpiexec -np 10 python -m mpi4py 'run-random-gYrE.py'
mpiexec -np 10 python -m mpi4py 'run-random-rYgE.py'
mpiexec -np 10 python -m mpi4py 'run-random-rYmE.py'
mpiexec -np 10 python -m mpi4py 'run-random-mYrE.py'
mpiexec -np 10 python -m mpi4py 'run-random-gYmE.py'
mpiexec -np 10 python -m mpi4py 'run-random-mYgE.py'
