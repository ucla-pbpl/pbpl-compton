import argparse
import toml

filename = 'run-random'
extension='py'
placeholder = "taggy"
options = ['image', 'triag', 'gYgE', 'mYmE', 'gYrE', 'rYgE', 'rYmE', 'mYrE', 'gYmE', 'mYgE']
parser = argparse.ArgumentParser(
        description='Run geant4 simulation based on config')
parser.add_argument("--config", required=True,
        help="path to the configuration toml file")
args = parser.parse_args()
conf = toml.load(args.config)
num_simulations = conf['Simulation']['NumSimulations']
for option in options:
    with open(filename+'-template'+"."+extension) as fp:
        outfilename = filename+'-'+option+"."+extension
        with open(outfilename, 'w' ) as fo:
            line = fp.readline()
            while line:
                line = line.replace(placeholder, option)
                fo.write(line)
                line = fp.readline()

with open('run_genres.sh', 'w') as bf:
    bf.write('set -x\n')
    bf.write('for i in {1..10}\n')
    bf.write('do\n')

    for option in options:
        bf.write("mpiexec -np {} python -m mpi4py \'run-random-{}.py\' --config {}\n".format(num_simulations, option, args.config))
    
    bf.write('sleep 1h\n')
    bf.write('done\n')
    