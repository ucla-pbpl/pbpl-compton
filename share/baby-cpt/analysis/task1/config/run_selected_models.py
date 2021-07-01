models_to_run = ["models.grid-Emory-Lynn-Rose-Adam-Bella-NABBEFELD"
, "models.grid-Emory-Lynn-Rose-Alice-Bella-NABERS"
, "models.grid-Emory-Lynn-Rose-Adam-Bella-NABB"
, "models.grid-Emory-Lynn-Rose-Adam-Beatrice-NABB"
, "models.grid-Emory-Lynn-Rose-Adam-Beatrice-NABARRETE"
, "models.grid-Emory-Lynn-Rose-Adam-Beatrice-NABBEFELD"
, "models.grid-Emory-Lynn-Rose-Adam-Beatrice-NABEL"
, "models.grid-Emory-Lynn-Rose-Adam-Bella-NABERHAUS"
, "models.grid-Emory-Lynn-Rose-Adam-Beatrice-NABA"
, "models.grid-Emory-Lynn-Rose-Adam-Beatrice-NABERS"
, "models.grid-Emory-Lynn-Rose-Alice-Bella-NABEL"
, "models.grid-Emory-Lynn-Rose-Alice-Beatrice-NABB"
, "models.grid-Emory-Lynn-Rose-Alice-Beatrice-NABERS"
, "models.grid-Emory-Lynn-Rose-Alice-Bella-NABA"
, "models.grid-Emory-Lynn-Rose-Alice-Bella-NABERHAUS"
, "models.grid-Emory-Lynn-Rose-Adam-Bella-NABARRETE"
, "models.grid-Emory-Lynn-Rose-Adam-Beatrice-NABARRO"
, "models.grid-Emory-Lynn-Rose-Alice-Beatrice-NABBEFELD"
, "models.grid-Emory-Lynn-Rose-Adam-Bella-NABERS"
, "models.grid-Emory-Lynn-Rose-Alice-Beatrice-NABI"
, "models.grid-Emory-Lynn-Rose-Alice-Beatrice-NABER"
, "models.grid-Emory-Lynn-Rose-Adam-Bella-NABEL"
, "models.grid-Emory-Lynn-Rose-Adam-Beatrice-NABHOLZ"
, "models.grid-Emory-Lynn-Rose-Adam-Beatrice-NABER"
, "models.grid-Emory-Lynn-Rose-Alice-Beatrice-NABA"
, "models.grid-Emory-Lynn-Rose-Alice-Bella-NABBEFELD"]

with open('run_selected_search.sh', 'w') as bf:
    bf.write('set -x\n')

    for st in models_to_run:
        bf.write("python ../train/grid_search_controller.py --config config/task1-{}.toml\n".format(st[12:]))
    
